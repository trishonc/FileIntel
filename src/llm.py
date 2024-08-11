from mlx_lm import load, generate
from prompt import SYSTEM_PROMPT
import json
import re

prompt_template = """<start_of_turn>user
{}
<end_of_turn>
<start_of_turn>model
"""

def call_agent(query, model, tokenizer):
    response = initial_call(query, model, tokenizer)
    parsed_response = parse_response(response)
    if parsed_response["type"] == "tool_call":
        execute_tool(parsed_response)
    else:
        print(parsed_response["content"])


def initial_call(query, model, tokenizer):
    prompt = prompt_template.format(SYSTEM_PROMPT+query)
    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=100)
    return response


def parse_response(response):
    response = response.replace("<end_of_turn>", "").strip()

    json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    
    if json_match:
        json_content = json_match.group(1)
        try:
            parsed_json = json.loads(json_content)

            tool_name = parsed_json.get('tool')
            attributes = parsed_json.get('attributes', {})

            return {
                'type': 'tool_call',
                'tool': tool_name,
                'attributes': attributes
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {
                'type': 'normal_response',
                'content': response
            }
    else:
        return {
            'type': 'normal_response',
            'content': response
        }



def execute_tool(parsed_response):
    print("Executing tool...")


