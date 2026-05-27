from huggingface_hub import InferenceClient

import os
API_KEY = os.getenv("HUGGINGFACE_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=API_KEY,
)

kk

embeddings = client.feature_extraction(
    text="Hello world",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("API Key Working ✅")
print(type(embeddings))
print(len(embeddings))