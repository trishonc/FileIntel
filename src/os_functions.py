import shutil
import os
from datetime import datetime
from typing import Dict


def os_move_file(srcPath: str, targetPath: str) -> Dict:
    try:
        os.makedirs(os.path.dirname(targetPath), exist_ok=True)
        
        shutil.move(srcPath, targetPath)
        print(f"File successfully moved from {srcPath} to {targetPath}")
        
        if os.path.isdir(targetPath):
            actual_target = os.path.join(targetPath, os.path.basename(srcPath))
        else:
            actual_target = targetPath

        stat = os.stat(actual_target)
        
        file_info = {
            "id": stat.st_ino, 
            "name": os.path.basename(actual_target),
            "path": actual_target,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size": stat.st_size
        }
        
        return file_info
    
    except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None


def os_copy_file(srcPath: str, targetPath: str) -> Dict:
    try:
        og_stat = os.stat(srcPath)
        og_id = og_stat.st_ino
        
        if os.path.isdir(targetPath):
            targetPath = os.path.join(targetPath, os.path.basename(srcPath))
        
        if os.path.samefile(srcPath, targetPath):
            base, ext = os.path.splitext(os.path.basename(srcPath))
            counter = 1
            while True:
                new_name = f"{base}_copy{counter}{ext}"
                targetPath = os.path.join(os.path.dirname(targetPath), new_name)
                if not os.path.exists(targetPath):
                    break
                counter += 1
        else:
            os.makedirs(os.path.dirname(targetPath), exist_ok=True)
        
        shutil.copy2(srcPath, targetPath)
        print(f"File successfully copied from {srcPath} to {targetPath}")
        
        new_stat = os.stat(targetPath)
        file_info = {
            "id": new_stat.st_ino,
            "og_id": og_id,
            "name": os.path.basename(targetPath),
            "path": targetPath,
            "created": datetime.fromtimestamp(new_stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(new_stat.st_mtime).isoformat(),
            "size": new_stat.st_size
        }
        return file_info
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def os_delete_file(filePath: str):
    try:
        os.remove(filePath)
        print(f"File {filePath} successfully deleted")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def os_rename_file(srcPath: str, newName: str) -> Dict:
    try:
        targetPath = os.path.join(os.path.dirname(srcPath), newName)
        os.rename(srcPath, targetPath)
        print(f"File successfully renamed from {srcPath} to {targetPath}")
        
        stat = os.stat(targetPath)
        file_info = {
            "id": stat.st_ino,
            "name": os.path.basename(targetPath),
            "path": targetPath,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size": stat.st_size
        }
        return file_info
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def os_open_file(file_path: str):
    try:
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        exit_status = os.system(f"open '{file_path}'")
        
        if exit_status == 0:
            print(f"File {file_path} successfully opened with the default application")
            return True
        else:
            raise Exception(f"Failed to open file. Exit status: {exit_status}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def os_goto_file(path: str):
    try:
        path = os.path.normpath(os.path.expanduser(path))
        
        if not os.path.exists(path):
            print(f"Error: The path '{path}' does not exist.")
            return

        if os.path.isfile(path):
            path = os.path.dirname(path)

        os.system(f"open '{path}'")
        
        print(f"Navigated to: {path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")