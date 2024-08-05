import argparse
from qdrant_client import QdrantClient, models
from query import parse_query
from update import update
from transformers import AutoModel
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Qdrant vector database script")
    parser.add_argument('-d', '--dirs', nargs='+', help='Directories to process', default=[])
    parser.add_argument('-r', '--recreate', action='store_true', help='Recreate the database')
    args = parser.parse_args()

    client = QdrantClient(path="/tmp/qdrant/db")

    if args.recreate or not client.collection_exists(collection_name="files"):
        if args.recreate and client.collection_exists(collection_name="files"):
            client.delete_collection(collection_name="files")
            print("Existing collection deleted.")

        client.create_collection(
            collection_name="files",
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )
        print("New collection created.")

    print("Loading model...")
    device = get_device()
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)

    print("Updating...")
    dirs = args.dirs if args.dirs else []
    update(client, dirs, model)

    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        parse_query(client, query, model)

if __name__ == "__main__":
    main()