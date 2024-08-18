from datasets import load_dataset
import random

def load_and_transform_dataset(dataset_name, transform_function=None, num_examples=None, max_examples=None):
    if max_examples:
        dataset = load_dataset(dataset_name, split=f'train[:{max_examples}]')
    else:
        dataset = load_dataset(dataset_name, split='train')
    
    if transform_function:
        dataset = dataset.map(transform_function)
    
    if num_examples:
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        dataset = dataset.select(indices)
    
    return dataset
