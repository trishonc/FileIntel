from llm.prompt import *
from globals import llm


def initial_call(query, tool_list):
    prompt = SYSTEM_PROMPT.format(tool_list, query)
    response = llm.invoke(prompt)
    return response["choices"][0]["text"]


def rag_call(query, context):
    prompt = RAG_PROMPT.format(context, query)
    response_generator = llm.stream(prompt)
    return response_generator