from tools import *
from qdrant_client import QdrantClient
from typing import List
from transformers import AutoModel
from utils import print_usage_instructions


def clean_query(query: str, keywords: List[str]):
    for keyword in keywords:
        query = query.replace(keyword, "").strip()
    return query.strip()


def parse_query(client: QdrantClient, query: str, llm, tokenizer, model: AutoModel):
    query = query.lower()
    query_parts = query.split()
    
    if query_parts[0] == "open":
        cleaned_query = clean_query(query, ["open"])
        open(client, cleaned_query, model)
    elif query_parts[0] == "move" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["move"])
        target_query = clean_query(query_parts[1], [])
        move_file(client, src_query, target_query, model)
    elif query_parts[0] == "copy" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["copy"])
        target_query = clean_query(query_parts[1], [])
        copy_file(client, src_query, target_query, model)
    elif query_parts[0] == "rename" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["rename"])
        new_name = clean_query(query_parts[1], [])
        rename_file(client, src_query, new_name, model)
    elif query_parts[0] == "go":
        cleaned_query = clean_query(query, ["go to"])
        goto_file(client, cleaned_query, model)
    elif query_parts[0] == "delete":
        cleaned_query = clean_query(query, ["delete"])
        delete_file(client, cleaned_query, model)
    elif "?" in query:
        local_search(client, query, llm, tokenizer, model)
    else:
        print_usage_instructions()