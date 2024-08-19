from datasets import load_dataset
import random

def load_and_transform_dataset(dataset_name, transform_function=None, num_examples=None, max_examples=None, remove_columns=True):
    if max_examples:
        dataset = load_dataset(dataset_name, split=f'train[:{max_examples}]')
    else:
        dataset = load_dataset(dataset_name, split='train')
    
    if transform_function:
        if remove_columns:
            dataset = dataset.map(transform_function, remove_columns=dataset.column_names)
        else:
            dataset = dataset.map(transform_function)


    if num_examples:
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        dataset = dataset.select(indices)
    
    return dataset


def tool_call_format(system, tools, query, response):
    text = f"""<start_of_turn>user
                {system}
                <tools>
                {tools}
                </tools>
                {query}
                <end_of_turn>
                <start_of_turn>model
                {response}
            """
    return text


def rag_format(system, context, query, response):
    text = f"""<start_of_turn>user
                {system}
                <context>
                {context}
                </context>
                {query}
                <end_of_turn>
                <start_of_turn>model
                {response}
            """
    return text


def general_format(system, query, response):
    text = f"""<start_of_turn>user
                {system}
                {query}
                <end_of_turn>
                <start_of_turn>model
                {response}
            """
    return text