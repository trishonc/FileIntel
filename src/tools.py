from search import vec_search, rag_search
from qdrant_client import QdrantClient
from transformers import AutoModel
from os_functions import *
from update import move_items, remove_items, copy_files
from llm.generate import rag_call
from utils import parse_response, confirm_operation


def open_file(client: QdrantClient, cleaned_query: str, model: AutoModel):
    file = vec_search(client, cleaned_query, model).payload["path"]
    os_open_file(file)


def move_file(client: QdrantClient, src_query: str, target_query: str, model: AutoModel):
    src = vec_search(client, src_query, model)
    target = vec_search(client, target_query, model)
    
    if "dir" in target.payload["type"]:
        if confirm_operation(f"Are you sure you want to move '{src.payload['path']}' to '{target.payload['path']}'?"):
            new_file = os_move_file(src.payload["path"], target.payload["path"])
            if new_file:
                move_items(client, [new_file], model)
                print(f"File moved to {new_file}")
    else:
        print("Can't move file to another file. Use rename instead.")


def copy_file(client: QdrantClient, src_query: str, target_query: str, model: AutoModel):
    src = vec_search(client, src_query, model)
    target = vec_search(client, target_query, model)
    
    if "dir" not in target.payload["type"]:
        print("Can't copy file to another file.")
    elif confirm_operation(f"Are you sure you want to copy '{src.payload['path']}' to '{target.payload['path']}'?"):
        new_file = os_copy_file(src.payload["path"], target.payload["path"])
        if new_file:
            copy_files(client, [new_file], model)
            print(f"File copied to {new_file}")


def rename_file(client: QdrantClient, src_query: str, new_name: str, model: AutoModel):
    src = vec_search(client, src_query, model)
    
    if confirm_operation(f"Are you sure you want to rename '{src.payload['path']}' to '{new_name}'?"):
        new_file = os_rename_file(src.payload["path"], new_name)
        if new_file:
            move_items(client, [new_file], model)
            print(f"File renamed to {new_file}")


def goto_file(client: QdrantClient, cleaned_query: str, model: AutoModel):
    file = vec_search(client, cleaned_query, model)
    os_goto_file(file.payload["path"])


def delete_file(client: QdrantClient, cleaned_query: str, model: AutoModel):
    file = vec_search(client, cleaned_query, model)
    file_path = file.payload["path"]
    
    if confirm_operation(f"Are you sure you want to delete '{file_path}'?"):
        os_delete_file(file_path)
        remove_items(client, [file.payload])
        print(f"File '{file_path}' has been deleted.")


def local_search(client: QdrantClient, query: str, llm, tokenizer, model: AutoModel):
    context = ""
    chunks = rag_search(client, query, model)
    for chunk in chunks:
        context += chunk.payload["content"]
    response = parse_response(rag_call(query, context, llm, tokenizer))
    print(response["content"])

