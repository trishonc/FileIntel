from utils import load_and_transform_dataset


system_prompt = "You are an ai assistant that has to a set of tools that you may use to help the user. Only use them if the user query requires them. For each tool call return a json object with the name of the tool and its arguments."


def reformat_data(examples):
    conversation = examples["conversations"]
    tool_list = examples["tools"]
       
    user = conversation[0]["value"]
    response = conversation[1]["value"]

    formatted_data = {
        "type": "tool_call",
        "system": system_prompt,
        "user": user,
        "tool_list": tool_list,
        "context": "",
        "response": response
    }
    
    return formatted_data


dataset = load_and_transform_dataset("llamafactory/glaive_toolcall_en", transform_function=reformat_data, max_examples=500, remove_columns=True)
dataset = dataset.rename_column("tool_list", "tools")
dataset.to_csv("tool_call_dataset.csv")