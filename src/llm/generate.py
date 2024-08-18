from llm.prompt import *
from tools import *
from globals import llm


def initial_call(query):
    prompt = SYSTEM_PROMPT + query
    response = llm.invoke(prompt)
    return response["choices"][0]["text"]


def rag_call(query, context):
    prompt = RAG_PROMPT.format(context, query)
    response_generator = llm.stream(prompt)
    return response_generator