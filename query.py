from search import vec_search
from tools import open_file, move_file, copy_file, rename_file, delete_file, goto_file
from qdrant_client import QdrantClient
from typing import List
from update import move_items, remove_items, copy_files
from transformers import AutoModel

def clean_query(query: str, keywords: List[str]):
    for keyword in keywords:
        query = query.replace(keyword, "").strip()
    return query.strip()

def parse_query(client: QdrantClient, query: str, model: AutoModel):
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
            new_file = move_file(src.payload["path"], target.payload["path"])
            if new_file:
                move_items(client, [new_file], model)
        else:
            print("Can't move file to another file. Use rename instead.")

    elif query.split()[0] == "copy" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["copy"])
        target_query = clean_query(query_parts[1], [])
        src = vec_search(client, src_query, model)
        target = vec_search(client, target_query, model)
       
        if "dir" not in target.payload["type"]:
            print("Cant copy file to another file.")

        else:     
            new_file = copy_file(src.payload["path"], target.payload["path"])
            if new_file:
                copy_files(client, [new_file], model)

    elif query.split()[0] == "rename" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["rename"])
        new_name = clean_query(query_parts[1], [])
        src = vec_search(client, src_query, model)

        new_file = rename_file(src.payload["path"], new_name)
        if new_file:
            move_items(client, [new_file], model)

    elif query.split()[0] == "go":
        cleaned_query = clean_query(query, ["go to"])
        file = vec_search(client, cleaned_query, model)

        goto_file(file.payload["path"])

    elif query.split()[0] == "delete":
        cleaned_query = clean_query(query, ["delete"])
        file = vec_search(client, cleaned_query, model)

        delete_file(file.payload["path"])
        remove_items(client, [file.payload])

    else:
        print("Unsupported operation or invalid query format.")
