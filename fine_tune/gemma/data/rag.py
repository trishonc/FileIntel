from utils import load_and_transform_dataset


system_prompt = "You are a helpful AI assistant and you should answer the user's query based on the provided context. Try to answer briefly and clearly."


def reformat_data(examples):
    context = examples["context"]
    user = examples["question"]
    response = examples["answer"]


    formatted_data = {
        "type": "rag",
        "system": system_prompt,
        "user": user,
        "tools": "",
        "ctx": context,
        "response": response
    }
    
    return formatted_data


dataset = load_and_transform_dataset("neural-bridge/rag-dataset-12000", transform_function=reformat_data, max_examples=500, remove_columns=True)
dataset = dataset.rename_column("ctx", "context")
dataset.to_csv("rag_dataset.csv")