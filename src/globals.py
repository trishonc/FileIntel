from qdrant_client import QdrantClient
from transformers import AutoModel
from utils import get_device
from langchain_ollama import ChatOllama


client = None
embedding_model = None
llm = None


def initialize_globals():
    global client, embedding_model, llm
    if client is None:
        client = QdrantClient(path="/tmp/qdrant/db")
    if embedding_model is None:
        print("Loading embedding model...")
        device = get_device()
        embedding_model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)
    if llm is None:
        print("Loading llm...")
        llm = ChatOllama(model="gemma2:2b")


initialize_globals()