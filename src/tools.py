from search import vec_search, rag_search
from os_functions import *
from update import move_items, remove_items, copy_files
from utils import confirm_operation
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field


class TargetOnlyInput(BaseModel):
    target: str = Field(description="should be the target file")


class RenameInput(BaseModel):
    source: str = Field(description="should be the source file")
    new_name: str = Field(description="should be the new name for the file")


class SourceAndTargetInput(BaseModel):
    source: str = Field(description="should be the source file")
    target: str = Field(description="should be the target location")


class LocalSearchInput(BaseModel):
    query: str = Field(description="should be the search query")


@tool(args_schema=TargetOnlyInput)
def open_file(target: str):
    """Open a file based on the given query."""
    file = vec_search(target).payload["path"]
    os_open_file(file)


@tool(args_schema=SourceAndTargetInput)
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


@tool(args_schema=SourceAndTargetInput)
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


@tool(args_schema=RenameInput)
def rename_file(source: str, new_name: str):
    """Rename a file."""
    source = vec_search(source)
    old_file_path = source.payload["path"]
    old_file_name, old_file_extension = os.path.splitext(old_file_path)
    if not os.path.splitext(new_name)[1]:  
        new_name += old_file_extension 
    if confirm_operation(f"Are you sure you want to rename '{old_file_path}' to '{new_name}'?"):
        new_file = os_rename_file(old_file_path, new_name)
        if new_file:
            move_items([new_file])
            return f"File renamed to {new_file}"


@tool(args_schema=TargetOnlyInput)
def goto_file(target: str):
    """Navigate to a file location."""
    file = vec_search(target)
    os_goto_file(file.payload["path"])


@tool(args_schema=TargetOnlyInput)
def delete_file(target: str):
    """Delete a file."""
    file = vec_search(target)
    file_path = file.payload["path"]
    if confirm_operation(f"Are you sure you want to delete '{file_path}'?"):
        os_delete_file(file_path)
        remove_items([file.payload])
        return f"File '{file_path}' has been deleted."


@tool(args_schema=LocalSearchInput)
def file_search(query: str):
    """Search for information in local files and documents."""
    from llm.generate import rag_call
    context = ""
    chunks = rag_search(query)
    for chunk in chunks:
        context += chunk.payload["content"]
    response_generator = rag_call(query, context)
    for chunk in response_generator:
        print(chunk["choices"][0]["text"], end="", flush=True)