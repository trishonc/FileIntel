from llama_cpp import Llama
from llm.prompt import PROMPT_TEMPLATE


class LLM:
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, verbose=False, n_ctx=8192, kv_overrides={"cache": True})

    def _generate(self, query, stream=False):
        prompt = PROMPT_TEMPLATE.format(query)
        return self.model(
            prompt,
            max_tokens=1000,
            stop=["<end_of_turn>"],
            stream=stream
        )

    def invoke(self, query):
        return self._generate(query)

    def stream(self, query):
        return self._generate(query, stream=True)
            