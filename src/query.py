from tools import *
from typing import List
from utils import print_usage_instructions


def clean_query(query: str, keywords: List[str]):
    for keyword in keywords:
        query = query.replace(keyword, "").strip()
    return query.strip()


def parse_query(query: str):
    query = query.lower()
    query_parts = query.split()
    
    if query_parts[0] == "open":
        cleaned_query = clean_query(query, ["open"])
        open_file(cleaned_query)
    elif query_parts[0] == "move" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["move"])
        target_query = clean_query(query_parts[1], [])
        move_file(src_query, target_query)
    elif query_parts[0] == "copy" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["copy"])
        target_query = clean_query(query_parts[1], [])
        copy_file(src_query, target_query)
    elif query_parts[0] == "rename" and " to " in query:
        query_parts = query.split(" to ")
        src_query = clean_query(query_parts[0], ["rename"])
        new_name = clean_query(query_parts[1], [])
        rename_file(src_query, new_name)
    elif query_parts[0] == "go":
        cleaned_query = clean_query(query, ["go to"])
        goto_file(cleaned_query)
    elif query_parts[0] == "delete":
        cleaned_query = clean_query(query, ["delete"])
        delete_file(cleaned_query)
    elif "?" in query:
        local_search(query)
    else:
        print_usage_instructions()