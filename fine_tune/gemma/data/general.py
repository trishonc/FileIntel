from utils import load_and_transform_dataset


system_prompt = "You are a helpful AI assistant and you answer the user's questions accurately and clearly."

columns_order = ["type", "system", "user", "tools", "context", "response"] 

def reformat_dolly_dataset(examples):
    context = examples["context"]
    user = examples["instruction"]
    response = examples["response"]
    category = examples["category"]


    formatted_data = {
        "type": category,
        "system": system_prompt,
        "user": user,
        "tools": "",
        "context": context,
        "response": response
    }
    
    return formatted_data


def reformat_ms_dataset(examples):
    user = examples["question"]
    response = examples["answer"]


    formatted_data = {
        "type": "reasoning",
        "system": system_prompt,
        "user": user,
        "tools": "",
        "context": "",
        "response": response
    }
    
    return formatted_data


def reformat_mp_dasaset(examples):
    user = examples["instruction"]
    response = examples["response"]


    formatted_data = {
        "type": "reasoning",
        "system": system_prompt,
        "user": user,
        "tools": "",
        "context": "",
        "response": response
    }
    
    return formatted_data


def reformat_ft_dataset(examples):
    conversation = examples["conversations"]
       
    user = conversation[0]["value"]
    response = conversation[1]["value"]

    formatted_data = {
        "type": "instruct",
        "system": system_prompt,
        "user": user,
        "tools": "",
        "context": "",
        "response": response
    }
    
    return formatted_data


dolly_dataset = load_and_transform_dataset("databricks/databricks-dolly-15k", transform_function=reformat_dolly_dataset, max_examples=500, remove_columns=True)
dolly_dataset = dolly_dataset.select_columns(columns_order)
# dolly_dataset.to_csv("dolly_dataset.csv")

ms_dataset = load_and_transform_dataset("microsoft/orca-math-word-problems-200k", transform_function=reformat_ms_dataset, max_examples=250, remove_columns=True)
ms_dataset = ms_dataset.select_columns(columns_order)
# ms_dataset.to_csv("ms_dataset.csv")

mp_dataset = load_and_transform_dataset("Magpie-Align/Magpie-Reasoning-150K", transform_function=reformat_mp_dasaset, max_examples=200, remove_columns=True)
mp_dataset = mp_dataset.select_columns(columns_order)
# mp_dataset.to_csv("mp_dataset.csv")

ft_dataset = load_and_transform_dataset("mlabonne/FineTome-100k", transform_function=reformat_ft_dataset, max_examples=250, remove_columns=True)
ft_dataset = ft_dataset.select_columns(columns_order)
# ft_dataset.to_csv("ft_dataset.csv")
