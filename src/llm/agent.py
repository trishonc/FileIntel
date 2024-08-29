from tools import *
from llm.generate import initial_call
from utils import parse_response
from langchain_core.utils.function_calling import convert_to_openai_tool
import json


tool_map = {
    "open_file": open_file,
    "goto_file": goto_file,
    "move_file": move_file,
    "copy_file": copy_file,
    "rename_file": rename_file,
    "delete_file": delete_file,
    "local_search": local_search
}

tools = [open_file, goto_file, move_file, copy_file, rename_file, delete_file, local_search]


def format_tools(tools):
    tool_list = []
    for tool in tools:
        tool = convert_to_openai_tool(tool)
        tool_list.append(tool)
    
    return json.dumps(tool_list)


def call_agent(query):
    tool_list = format_tools(tools)
    response = initial_call(query, tool_list)
    # print(response)
    parsed_response = parse_response(response)
    # print(parsed_response)
    if parsed_response["type"] == "tool_call":
        execute_tool(parsed_response)
    else:
        print(parsed_response["content"])


def execute_tool(parsed_response):
    tool = parsed_response["name"]
    args = parsed_response["arguments"]
    if tool in tool_map:
        try:
            tool_map[tool].invoke(args)
        except TypeError as e:
            print(f"Error executing tool '{tool}': {str(e)}")
            print("Please check the arguments provided.")
        except Exception as e:
            print(f"An error occurred while executing tool '{tool}': {str(e)}")
    else:
        print(f"Invalid tool name - {tool}")
