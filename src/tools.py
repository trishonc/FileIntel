from search import vec_search, rag_search
from os_functions import *
from update import move_items, remove_items, copy_files
from utils import confirm_operation


def open_file(target: str):
    """Open a file based on the given query."""
    file = vec_search(target).payload["path"]
    os_open_file(file)


def move_file(source: str, target: str):
    """Move a file from source to target location."""
    source = vec_search(source)
    target = vec_search(target)
    if "dir" in target.payload["type"]:
        if confirm_operation(f"Are you sure you want to move '{source.payload['path']}' to '{target.payload['path']}'?"):
            new_file = os_move_file(source.payload["path"], target.payload["path"])
            if new_file:
                move_items([new_file])
                return f"File moved to {new_file}"
    else:
        return "Can't move file to another file. Use rename instead."


def copy_file(source: str, target: str):
    """Copy a file from source to target location."""
    source = vec_search(source)
    target = vec_search(target)
    if "dir" not in target.payload["type"]:
        return "Can't copy file to another file."
    elif confirm_operation(f"Are you sure you want to copy '{source.payload['path']}' to '{target.payload['path']}'?"):
        new_file = os_copy_file(source.payload["path"], target.payload["path"])
        if new_file:
            copy_files([new_file])
            return f"File copied to {new_file}"


def rename_file(source: str, new_name: str):
    """Rename a file."""
    source = vec_search(source)
    if confirm_operation(f"Are you sure you want to rename '{source.payload['path']}' to '{new_name}'?"):
        new_file = os_rename_file(source.payload["path"], new_name)
        if new_file:
            move_items([new_file])
            return f"File renamed to {new_file}"


def goto_file(target: str):
    """Navigate to a file location."""
    file = vec_search(target)
    os_goto_file(file.payload["path"])


def delete_file(target: str):
    """Delete a file."""
    file = vec_search(target)
    file_path = file.payload["path"]
    if confirm_operation(f"Are you sure you want to delete '{file_path}'?"):
        os_delete_file(file_path)
        remove_items([file.payload])
        return f"File '{file_path}' has been deleted."


def local_search(query: str):
    """Perform a local search using RAG."""
    from llm.generate import rag_call
    context = ""
    chunks = rag_search(query)
    for chunk in chunks:
        context += chunk.payload["content"]
    response_generator = rag_call(query, context)
    for chunk in response_generator:
        print(chunk["choices"][0]["text"], end="", flush=True)