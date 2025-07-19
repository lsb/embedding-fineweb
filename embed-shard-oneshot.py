# This runs a particular fineweb shard's embeddings

# ! pip install "accelerate"  "huggingface_hub[hf_transfer]" "datasets" "numpy" "torch" "triton" "sentence-transformers" "transformers" "optimum-quanto"

import sys
import os
# os.environ["HF_TOKEN"] = "your-token-here"


import torch
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import repo_exists
from sentence_transformers import SentenceTransformer

if True:
    EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    short_name = "paraphrase-multilingual-minilml12v2"
else:
    EMB_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    short_name = "qwen3"

#device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


fw = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
fw = fw.with_format("numpy")


def embed(b):
    with torch.no_grad():
        enc = emb.encode(b['text'], show_progress_bar=False, device=device)
        b[short_name] = enc
        return b
    
if __name__ == "__main__":
    # we will be called like: python (thisfile.py) gpu_number batch_size shard1 shard2 ...
    # and we need to process each shard in turn
    gpu_number = int(sys.argv[1])
    device = f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer(EMB_MODEL_NAME, device=device, model_kwargs={"torch_dtype": dtype}).eval()
    emb = torch.compile(emb, mode="max-autotune")

    batch_size = int(sys.argv[2])
    shard_ids = [int(s) for s in sys.argv[3:]]
    for shard_id in shard_ids:
        repo = f"lsb/fineweb-{short_name}-shard-{shard_id}"
        if repo_exists(repo, repo_type="dataset"):
            print(f"Skipping shard {shard_id} as it already exists in the hub.")
            continue
        print(f"Processing shard {shard_id} with batch size {batch_size}")
        batches = []
        for batch in tqdm(fw.shard(fw.num_shards, shard_id).map(embed, batched=True, batch_size=batch_size).batch(batch_size=batch_size)):
            d = Dataset.from_dict({"id": batch['id'], short_name: batch[short_name]})
            batches.append(d)
        concatenate_datasets(batches).push_to_hub(repo)




