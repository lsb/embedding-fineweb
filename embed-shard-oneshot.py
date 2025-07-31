# This runs a particular fineweb shard's embeddings

# ! pip install "accelerate"  "huggingface_hub[hf_transfer]" "datasets" "numpy" "torch" "triton" "sentence-transformers" "transformers" "optimum-quanto"

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.client import Config

import torch
from tqdm import tqdm
import random
import datetime
from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

if True:
    EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    short_name = "paraphrase-multilingual-minilml12v2"
    test_batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
else:
    EMB_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    short_name = "qwen3"
    test_batch_sizes = [1, 2, 4]

#device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


fw = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
fw = fw.with_format("numpy")
bucket_name = os.environ['BUCKET_NAME']
dir = "fineweb-shard"

def shard_id_to_lock_name(shard_id):
    """Convert shard ID to a lock file name."""
    return f"{dir}/{shard_id}.lock"


def shard_id_to_parquet_name(shard_id):
    """Convert shard ID to a parquet file name."""
    return f"{dir}/{shard_id}.parquet"


def list_all_keys(s3_client, page_size = 1_000):
    """
    Fetch *every* object key under `bucket/prefix` and return them in a list.
    Runs entirely in this thread; safe to call from a worker.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = {}
    for page in paginator.paginate(
            Bucket=bucket_name,
            Prefix=dir,
            PaginationConfig={"PageSize": page_size}):   # ≤1000 keys/page
        for obj in page.get("Contents", []):
            keys[obj["Key"]] = True
    return keys



if __name__ == "__main__":
    # we receive no commandline arguments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer(EMB_MODEL_NAME, device=device, model_kwargs={"torch_dtype": dtype}).eval()
    emb = torch.compile(emb, mode="max-autotune")
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3-lister")

    shard_ids = list(range(fw.num_shards))
    random.shuffle(shard_ids)
    shard_ids = [25164, 25165, 25166, 25167] + shard_ids

    def embed(b):
        with torch.no_grad():
            enc = emb.encode(b['text'], show_progress_bar=False, device=device)
            b[short_name] = torch.tensor(enc).to(torch.float8_e4m3fn).to(torch.float32).numpy()
            return b

    # Initialize S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    existing_keys = {}

    future = pool.submit(list_all_keys, s3)

    # Test batch sizes
    test_corpus = [" ".join([str(i) for i in range(lo, 20000)]) for lo in range(1024) ]
    test_dataset = Dataset.from_dict({"id": list(range(len(test_corpus))), "text": test_corpus})
    test_batch_size_timings = {}
    for batch_size in test_batch_sizes:
        print(f"Testing batch size {batch_size} on {device}")
        batches = []
        start_time = datetime.datetime.now()
        for batch in tqdm(test_dataset.map(embed, batched=True, batch_size=batch_size).batch(batch_size=batch_size)):
            d = Dataset.from_dict({"id": batch['id'], short_name: batch[short_name]})
            batches.append(d)
        final_dataset = concatenate_datasets(batches)
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        test_batch_size_timings[batch_size] = elapsed_time
        print(f"Batch size of {batch_size} took {elapsed_time:.2f} seconds")

    # choose the best batch size based on the fastest time
    best_batch_size = min(test_batch_size_timings, key=test_batch_size_timings.get)
    print(f"Best batch size: {best_batch_size} with time {test_batch_size_timings[best_batch_size]:.2f} seconds")

    existing_keys = future.result()

    for shard_id in shard_ids:
        # Check if parquet file exists in our S3 cache
        if shard_id_to_parquet_name(shard_id) in existing_keys:
            print(f"Skipping shard {shard_id} as we see it is already completed in S3 at s3://{bucket_name}/{shard_id_to_parquet_name(shard_id)}")
            continue
        # Try to fetch the lock file. Contents are when it should expire.
        try:
            lock_response = s3.get_object(Bucket=bucket_name, Key=shard_id_to_lock_name(shard_id))
            lock_contents = lock_response['Body'].read().decode('utf-8')
            lock_expiry = datetime.datetime.fromisoformat(lock_contents.strip())
            if lock_expiry > datetime.datetime.now():
                print(f"Skipping shard {shard_id} as it is locked until {lock_expiry} in S3 at s3://{bucket_name}/{shard_id_to_lock_name(shard_id)}")
                continue
            else:
                # write a new lock file
                print(f"Lock file for shard {shard_id} has expired, proceeding to create a new one.")
                # we write a new one below
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"No existing lock file for shard {shard_id}, proceeding to create one.")
                # only a tiny toctou because CAS is not for anonymous access, and double-processing should be idempotent, if a little pricy
            else:
                print(f"Error checking lock file for shard {shard_id}: {e}")
                raise e
        # Put the lock file in S3 with a 1-day expiration
        try:
            s3.put_object(
                Bucket=bucket_name,
                Key=shard_id_to_lock_name(shard_id),
                Body=f"{(datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()}".encode('utf-8'),
                # IfNoneMatch='*'  # This will raise an error if the object already exists
            )
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'PreconditionFailed':
                print(f"Skipping shard {shard_id} as we just saw it is locked in S3 at s3://{bucket_name}/{shard_id_to_lock_name(shard_id)}")
                continue  # Lock file already exists, skip this shard
            else:
                print(f"Error creating lock file for shard {shard_id}: {e}")
                raise e

        print(f"Processing shard {shard_id} with batch size {best_batch_size}")
        batches = []
        for batch in tqdm(fw.shard(fw.num_shards, shard_id).map(embed, batched=True, batch_size=best_batch_size).batch(batch_size=best_batch_size)):
            d = Dataset.from_dict({"id": batch['id'], short_name: batch[short_name]})
            batches.append(d)

        # Concatenate all batches and save to parquet
        final_dataset = concatenate_datasets(batches)

        future = pool.submit(list_all_keys, s3)

        # Save to local parquet file
        local_file = f"/tmp/shard.parquet"
        final_dataset.to_parquet(local_file)

        # Upload to S3
        print(f"Uploading shard {shard_id} to s3://{bucket_name}/{shard_id_to_parquet_name(shard_id)}")
        s3.put_object(
            Bucket=bucket_name,
            Key=shard_id_to_parquet_name(shard_id),
            Body=Path(local_file).read_bytes(),
        ) # no multipart uploads when anonymous

        # Clean up local file
        os.remove(local_file)
        existing_keys = future.result()
        print(f"Successfully uploaded shard {shard_id}")
