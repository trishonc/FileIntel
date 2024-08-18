from llm.prompt import *
from tools import *
from globals import llm


def initial_call(query):
    sys_prompt = SYSTEM_PROMPT + query
    prompt = PROMPT_TEMPLATE.format(sys_prompt)
    response = llm.invoke(prompt)
    return response["choices"][0]["text"]


def rag_call(query, context):
    rag_prompt = RAG_PROMPT.format(context, query)
    prompt = PROMPT_TEMPLATE.format(rag_prompt)
    response_generator = llm.stream(prompt)
    return response_generator