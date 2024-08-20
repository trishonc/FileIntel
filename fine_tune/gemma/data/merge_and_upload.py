from datasets import concatenate_datasets
from general import dolly_dataset, mp_dataset, ms_dataset, ft_dataset
from tool_call import tool_call_dataset
from rag import rag_dataset

dataset_list = [dolly_dataset, mp_dataset, ms_dataset, ft_dataset, tool_call_dataset, rag_dataset]

dataset = concatenate_datasets(dataset_list)
dataset.push_to_hub('trishonc/agent-buff')