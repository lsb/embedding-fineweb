FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages in a single layer
RUN pip install sentence-transformers transformers datasets huggingface_hub accelerate numpy tqdm boto3

# Pre-download the embedding model to cache it in the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
    print('Model downloaded successfully')"

# Copy the embedding script
COPY embed-shard-oneshot.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Entry point
ENTRYPOINT ["python", "embed-shard-oneshot.py"]
