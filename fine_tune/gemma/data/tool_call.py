from utils import load_and_transform_dataset


system_prompt = "You are an ai assistant that has to a set of tools that you may use to help the user. Only use them if the user query requires them. For each tool call return a json object with the name of the tool and its arguments."

columns_order = ["type", "system", "user", "tools", "context", "response"] 


def reformat_data(examples):
    conversation = examples["conversations"]
    tool_list = examples["tools"]
       
    user = conversation[0]["value"]
    response = conversation[1]["value"]
    response_type = conversation[1]["from"]


    formatted_data = {
        "type": "tool_call",
        "system": system_prompt,
        "user": user,
        "tools": tool_list if tool_list.strip() in ("[]", "") else f"<tools>\n{tool_list}\n</tools>",
        "context": "",
        "response": response if response_type == "gpt" else f"<tool_call>\n{response}\n</tool_call>"
    }
    
    return formatted_data


tool_call_dataset = load_and_transform_dataset("llamafactory/glaive_toolcall_en", transform_function=reformat_data, max_examples=500, remove_columns=True)
tool_call_dataset = tool_call_dataset.select_columns(columns_order)
tool_call_dataset.to_csv("tool_call_dataset.csv")