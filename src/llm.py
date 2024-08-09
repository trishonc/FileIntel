from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")

prompt_template = """<start_of_turn>user
{}
<end_of_turn>
<start_of_turn>model
"""

prompt = prompt_template.format("Hello")

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100)
