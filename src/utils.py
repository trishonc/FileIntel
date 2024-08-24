import re
import json
import torch 

def parse_response(response):
    response = response.replace("<end_of_turn>", "").strip()

    json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    
    if json_match:
        json_content = json_match.group(1)
        try:
            parsed_json = json.loads(json_content)

            tool_name = parsed_json.get('tool')
            args = parsed_json.get('arguments', {})

            return {
                'type': 'tool_call',
                'tool': tool_name,
                'arguments': args
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



def format_response(response):
    response = response.replace('/n', '\n')
    response = response.replace('*', '')
    return response


def confirm_operation(message: str) -> bool:
    confirmation = input(f"{message} (y/N): ").lower()
    return confirmation == 'y'


def print_usage_instructions():
    print("""Unsupported operation or invalid query format. Valid query formats are:
             open <target>
             go to <target>
             delete <target>
             rename <target> to <new_name>
             move <source> to <target>
             copy <source> to <target>
             <query> ?
          """)
    

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"