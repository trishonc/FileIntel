import argparse
from qdrant_client import models
from query import parse_query
from update import update
from llm.agent import call_agent
from globals import *


def main():
    parser = argparse.ArgumentParser(description="Qdrant vector database script")
    parser.add_argument('-d', '--dirs', nargs='+', help='Directories to process', default=[])
    parser.add_argument('-r', '--recreate', action='store_true', help='Recreate the database')
    parser.add_argument('--llm', action='store_true', help='Use LLM agent for querying')
    args = parser.parse_args()

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

    print("Updating...")
    dirs = args.dirs if args.dirs else []
    update(dirs)

    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        if args.llm:
            call_agent(query)
        else:
            parse_query(query)


if __name__ == "__main__":
    main()