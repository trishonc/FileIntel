from mlx_lm import generate
from prompt import *
import json
import re
from search import vec_search
from tools import *
from update import move_items, copy_files, remove_items

def call_agent(client, query, llm, tokenizer, embedding_model):
    response = initial_call(query, llm, tokenizer)
    parsed_response = parse_response(response)
    if parsed_response["type"] == "tool_call":
        execute_tool(client, parsed_response, llm, tokenizer, embedding_model)
    else:
        print(format_response(parsed_response["content"]))


def initial_call(query, llm, tokenizer):
    sys_prompt = SYSTEM_PROMPT + query
    prompt = PROMPT_TEMPLATE.format(sys_prompt)
    response = generate(llm, tokenizer, prompt=prompt, verbose=False, max_tokens=500)
    return response


def rag_call(query, context, llm, tokenizer):
    rag_prompt = RAG_PROMPT.format(context, query)
    prompt = PROMPT_TEMPLATE.format(rag_prompt)
    response = generate(llm, tokenizer, prompt=prompt, verbose=False, max_tokens=500)
    return response


def execute_tool(client, parsed_response, llm, tokenizer, embedding_model):
    tool = parsed_response["tool"]
    arg = parsed_response["args"]

    if tool == "open_file":
        file = vec_search(client, arg["target"], embedding_model).payload["path"]
        open_file(file)
    elif tool == "goto_file":
        file = vec_search(client, arg["target"], embedding_model).payload["path"]
        goto_file(file)
    elif tool == "move_file":
        src = vec_search(client, arg["src"], embedding_model)
        target = vec_search(client, arg["target"], embedding_model)
        if "dir" in target.payload["type"]:
            confirmation = input(f"Are you sure you want to move '{src.payload['path']}' to '{target.payload['path']}'? (y/N): ").lower()
            if confirmation == 'y':
                new_file = move_file(src.payload["path"], target.payload["path"])
                if new_file:
                    move_items(client, [new_file], embedding_model)
                    print(f"File moved to {new_file}")
            else:
                print("Move operation cancelled.")
        else:
            print("Can't move file to another file. Use rename instead.")
    elif tool == "copy_file":
        src = vec_search(client, arg["src"], embedding_model)
        target = vec_search(client, arg["target"], embedding_model)
        if "dir" not in target.payload["type"]:
            print("Can't copy file to another file.")
        else:
            confirmation = input(f"Are you sure you want to copy '{src.payload['path']}' to '{target.payload['path']}'? (y/N): ").lower()
            if confirmation == 'y':
                new_file = copy_file(src.payload["path"], target.payload["path"])
                if new_file:
                    copy_files(client, [new_file], embedding_model)
                    print(f"File copied to {new_file}")
            else:
                print("Copy operation cancelled.")
    elif tool == "rename_file":
        src = vec_search(client, arg["src"], embedding_model)
        confirmation = input(f"Are you sure you want to rename '{src.payload['path']}' to '{arg["new_name"]}'? (y/N): ").lower()
        if confirmation == 'y':
            new_file = rename_file(src.payload["path"], arg["new_name"])
            if new_file:
                move_items(client, [new_file], embedding_model)
                print(f"File renamed to {new_file}")
        else:
            print("Rename operation cancelled.")
    elif tool == "delete_file":
        file = vec_search(client, arg["target"], embedding_model)
        file_path = file.payload["path"]
        confirmation = input(f"Are you sure you want to delete '{file_path}'? (y/N): ").lower()
        if confirmation == 'y':
            delete_file(file_path)
            remove_items(client, [file.payload])
            print(f"File '{file_path}' has been deleted.")
        else:
            print("Deletion cancelled.")
    elif tool == "local_search":
        context = local_search(client, arg["query"], embedding_model)
        response = rag_call(arg["query"], context, llm, tokenizer)
        response = response.replace("<end_of_turn>", "").strip()   
        print(format_response(response))
    else:
        print(f"Invalid tool name - {tool}")


def parse_response(response):
    response = response.replace("<end_of_turn>", "").strip()

    json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    
    if json_match:
        json_content = json_match.group(1)
        try:
            parsed_json = json.loads(json_content)

            tool_name = parsed_json.get('tool')
            args = parsed_json.get('args', {})

            return {
                'type': 'tool_call',
                'tool': tool_name,
                'args': args
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {
                'type': 'normal_response',
                'content': response
            }
    else:
        return {
            'type': 'normal_response',
            'content': response
        }


def format_response(response):
    response = response.replace('/n', '\n')
    response = response.replace('*', '')
    return response