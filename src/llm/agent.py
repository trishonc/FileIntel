from tools import *
from llm.generate import initial_call
from utils import parse_response, format_response


def call_agent(client, query, llm, tokenizer, embedding_model):
    response = initial_call(query, llm, tokenizer)
    parsed_response = parse_response(response)
    if parsed_response["type"] == "tool_call":
        execute_tool(client, parsed_response, llm, tokenizer, embedding_model)
    else:
        print(format_response(parsed_response["content"]))


def execute_tool(client, parsed_response, llm, tokenizer, embedding_model):
    tool = parsed_response["tool"]
    arg = parsed_response["args"]

    tool_map = {
        "open_file": lambda: open_file(client, arg["target"], embedding_model),
        "goto_file": lambda: goto_file(client, arg["target"], embedding_model),
        "move_file": lambda: move_file(client, arg["source"], arg["target"], embedding_model),
        "copy_file": lambda: copy_file(client, arg["source"], arg["target"], embedding_model),
        "rename_file": lambda: rename_file(client, arg["source"], arg["new_name"], embedding_model),
        "delete_file": lambda: delete_file(client, arg["target"], embedding_model),
        "local_search": lambda: local_search(client, arg["query"], llm, tokenizer, embedding_model)
    }

    if tool in tool_map:
        tool_map[tool]()
    else:
        print(f"Invalid tool name - {tool}")
