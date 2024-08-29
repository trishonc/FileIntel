import re
import json
import torch 


def parse_response(response):
    response = response.replace("<end_of_turn>", "").strip()
    
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    if tool_call_match:
        tool_call_content = tool_call_match.group(1).strip()
        try:
            parsed_json = json.loads(tool_call_content)
            
            if isinstance(parsed_json, list):
                tool_data = parsed_json[0] if parsed_json else {}
            else:
                tool_data = parsed_json
            
            tool_name = tool_data.get('name')
            arguments = tool_data.get('arguments', {})
            
            return {
                'type': 'tool_call',
                'name': tool_name,
                'arguments': arguments
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")
        
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