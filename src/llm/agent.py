from tools import *
from llm.generate import initial_call
from utils import parse_response


tool_map = {
    "open_file": open_file,
    "goto_file": goto_file,
    "move_file": move_file,
    "copy_file": copy_file,
    "rename_file": rename_file,
    "delete_file": delete_file,
    "local_search": local_search
}


def call_agent(query):
    response = initial_call(query)
    parsed_response = parse_response(response)
    if parsed_response["type"] == "tool_call":
        execute_tool(parsed_response)
    else:
        print(parsed_response["content"])


def execute_tool(parsed_response):
    tool = parsed_response["tool"]
    args = parsed_response["args"]
    if tool in tool_map:
        try:
            tool_map[tool](**args)
        except TypeError as e:
            print(f"Error executing tool '{tool}': {str(e)}")
            print("Please check the arguments provided.")
        except Exception as e:
            print(f"An error occurred while executing tool '{tool}': {str(e)}")
    else:
        print(f"Invalid tool name - {tool}")
