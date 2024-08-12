from search import vec_search, rag_search
from tools import *
from qdrant_client import QdrantClient
from typing import List
from update import move_items, remove_items, copy_files
from transformers import AutoModel
from llm import rag_call, parse_response


def clean_query(query: str, keywords: List[str]):
    for keyword in keywords:
        query = query.replace(keyword, "").strip()
    return query.strip()


def parse_query(client: QdrantClient, query: str, llm, tokenizer, model: AutoModel):
    query = query.lower()
    if query.split()[0] == "open":
        cleaned_query = clean_query(query, ["open"])
        file = vec_search(client, cleaned_query, model).payload["path"]
        open_file(file)
    elif query.split()[0] == "move" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["move"])
        target_query = clean_query(query_parts[1], [])
        src = vec_search(client, src_query, model)
        target = vec_search(client, target_query, model)
        if "dir" in target.payload["type"]:
            confirmation = input(f"Are you sure you want to move '{src.payload['path']}' to '{target.payload['path']}'? (y/N): ").lower()
            if confirmation == 'y':
                new_file = move_file(src.payload["path"], target.payload["path"])
                if new_file:
                    move_items(client, [new_file], model)
                    print(f"File moved to {new_file}")
            else:
                print("Move operation cancelled.")
        else:
            print("Can't move file to another file. Use rename instead.")
    elif query.split()[0] == "copy" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["copy"])
        target_query = clean_query(query_parts[1], [])
        src = vec_search(client, src_query, model)
        target = vec_search(client, target_query, model)
        if "dir" not in target.payload["type"]:
            print("Can't copy file to another file.")
        else:
            confirmation = input(f"Are you sure you want to copy '{src.payload['path']}' to '{target.payload['path']}'? (y/N): ").lower()
            if confirmation == 'y':
                new_file = copy_file(src.payload["path"], target.payload["path"])
                if new_file:
                    copy_files(client, [new_file], model)
                    print(f"File copied to {new_file}")
            else:
                print("Copy operation cancelled.")
    elif query.split()[0] == "rename" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["rename"])
        new_name = clean_query(query_parts[1], [])
        src = vec_search(client, src_query, model)
        confirmation = input(f"Are you sure you want to rename '{src.payload['path']}' to '{new_name}'? (y/N): ").lower()
        if confirmation == 'y':
            new_file = rename_file(src.payload["path"], new_name)
            if new_file:
                move_items(client, [new_file], model)
                print(f"File renamed to {new_file}")
        else:
            print("Rename operation cancelled.")
    elif query.split()[0] == "go":
        cleaned_query = clean_query(query, ["go to"])
        file = vec_search(client, cleaned_query, model)
        goto_file(file.payload["path"])
    elif query.split()[0] == "delete":
        cleaned_query = clean_query(query, ["delete"])
        file = vec_search(client, cleaned_query, model)
        file_path = file.payload["path"]
        confirmation = input(f"Are you sure you want to delete '{file_path}'? (y/N): ").lower()
        if confirmation == 'y':
            delete_file(file_path)
            remove_items(client, [file.payload])
            print(f"File '{file_path}' has been deleted.")
        else:
            print("Deletion cancelled.")
    elif "?" in query:
        context = get_context(client, query, model)
        response = parse_response(rag_call(query, context, llm, tokenizer))
        print(response["content"])
    else:
        print("Unsupported operation or invalid query format.")