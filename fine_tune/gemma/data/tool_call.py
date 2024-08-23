from utils import load_and_transform_dataset
import json
import random


system_prompt = "You are an ai assistant that has to a set of tools that you may use to help the user. Only use them if the user query requires them. For each tool call return a json object with the name of the tool and its arguments."

columns_order = ["tools", "user", "response", "tool_call"] 


def add_tools(current_tools, extra_tools):
    max_tools = 10
    current_count = len(current_tools)
    max_to_add = max(0, max_tools - current_count)
    
    if max_to_add == 0:
        return current_tools
    
    tools_to_add_count = random.randint(1, max_to_add)
    
    selected_tools = []
    for tool_list in extra_tools:
        tool_list = json.loads(tool_list)
        if len(selected_tools) < tools_to_add_count:
            selected_tools.extend(random.sample(tool_list, min(len(tool_list), tools_to_add_count - len(selected_tools))))
        else:
            break
    
    addition_method = random.choice(["front", "back", "mix"])
    
    if addition_method == "front":
        return selected_tools + current_tools
    elif addition_method == "back":
        return current_tools + selected_tools
    else:  # mix
        combined = current_tools + selected_tools
        random.shuffle(combined)
        return combined


def reformat_data(examples, extra_tools):
    conversation = examples["conversations"]
    tools = examples["tools"]
    tools = json.loads(tools)

    if tools and random.random() < 0.5:
        new_tools = add_tools(tools, extra_tools)
        new_tools = json.dumps(new_tools)
    else:
        new_tools = json.dumps(tools)

    user = conversation[0]["value"]
    response = conversation[1]["value"]
    response_type = conversation[1]["from"]

    data = {
        "tools": new_tools,
        "user": user,
        "response": response,
        "tool_call": "yes" if response_type == "function_call" else "no"
    }

    return data


tool_call_dataset = load_and_transform_dataset("llamafactory/glaive_toolcall_en", transform_function=reformat_data, remove_columns=True)
tool_call_dataset = tool_call_dataset.select_columns(columns_order)
tool_call_dataset.to_csv("new_tool_call_dataset.csv")