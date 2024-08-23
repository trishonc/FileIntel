from datasets import load_dataset, Dataset
import random


def load_and_transform_dataset(dataset_name, transform_function=None, num_examples=None, max_examples=None, remove_columns=True):
    if max_examples:
        full_dataset = load_dataset(dataset_name, split=f'train[:{max_examples}]')
    else:
        full_dataset = load_dataset(dataset_name, split='train')

    half_index = len(full_dataset) // 2
    dataset = Dataset.from_dict(full_dataset[:half_index])
    extra_tools = full_dataset[half_index:]["tools"]
    
    
    if transform_function:
        if remove_columns:
            dataset = dataset.map(lambda x: transform_function(x, extra_tools), remove_columns=dataset.column_names)
        else:
            dataset = dataset.map(lambda x: transform_function(x, extra_tools))


    if num_examples:
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        dataset = dataset.select(indices)
    
    return dataset