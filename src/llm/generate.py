from mlx_lm import generate
from llm.prompt import *


def initial_call(query, llm, tokenizer):
    sys_prompt = SYSTEM_PROMPT + query
    prompt = PROMPT_TEMPLATE.format(sys_prompt)
    response = generate(llm, tokenizer, prompt=prompt, verbose=False, max_tokens=500)
    return response


def rag_call(query, context, llm, tokenizer):
    rag_prompt = RAG_PROMPT.format(context, query)
    prompt = PROMPT_TEMPLATE.format(rag_prompt)
    response = generate(llm, tokenizer, prompt=prompt, verbose=False, max_tokens=500)
    return response