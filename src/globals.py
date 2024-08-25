from qdrant_client import QdrantClient
from transformers import AutoModel
from utils import get_device
from llm.model import LLM
from huggingface_hub import hf_hub_download
import os


client = None
embedding_model = None
llm = None

repo_id = "bartowski/gemma-2-2b-it-abliterated-GGUF"
filename = "gemma-2-2b-it-abliterated-Q4_K_L.gguf"
local_dir = "."


def initialize_globals():
    global client, embedding_model, llm
    if client is None:
        client = QdrantClient(path="/tmp/qdrant/db")
    if embedding_model is None:
        print("Loading embedding model...")
        device = get_device()
        embedding_model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)
    if llm is None:
        if not os.path.exists(os.path.join(local_dir, filename)):
            print(f"Llm not found. Downloading from Hugging Face Hub...")
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            print(f"Download complete. File saved to {file_path}")
        else:
            file_path = os.path.join(local_dir, filename)
        print("Loading llm...")
        llm = LLM(file_path)


initialize_globals()