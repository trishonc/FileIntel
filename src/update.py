from qdrant_client import QdrantClient, models
from embed import add_all, embed_string
from typing import List, Dict, Tuple
import os
from datetime import datetime
import uuid
from transformers import AutoModel
from search import id_search
import json


def process_filesystem(root_dirs: List[str], include_files: bool = True, include_dirs: bool = True) -> List[Dict]:
    items = []
    
    allowed_file_extensions = {
        'image': ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp'),
        'pdf': ('.pdf',),
        'csv': ('.csv',),
        'doc': ('.doc', '.docx'),
        'ppt': ('.ppt', '.pptx'),
        'ppt': ('.ppt', '.pptx'),
    }
    
    with open('text-extensions.json', 'r') as f:
        text_file_extensions = set(json.load(f))

    def create_item_dict(path: str, item_type: str) -> Dict:
        stat = os.stat(path)
        item = {
            "id": stat.st_ino,
            "name": os.path.basename(path),
            "path": path,
            "type": item_type,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
        if item_type != "dir":
            item["size"] = stat.st_size
        return item

    def determine_file_type(filename: str) -> str:
        lower_filename = filename.lower()
        _, ext = os.path.splitext(lower_filename)
        

        for ftype, exts in allowed_file_extensions.items():
            if ext in exts:
                return ftype
        
        ext = ext.replace(".", "")
        if ext in text_file_extensions:
            return 'text'
        
        return None  

    for dir_path in root_dirs:
        if include_dirs:
            items.append(create_item_dict(dir_path, "dir"))

        for root, subdirs, filenames in os.walk(dir_path):
            if include_dirs:
                for subdir in subdirs:
                    items.append(create_item_dict(os.path.join(root, subdir), "dir"))

            if include_files:
                for filename in filenames:
                    file_path = os.path.join(root, filename)

                    file_type = determine_file_type(filename)
                    if file_type:
                        items.append(create_item_dict(file_path, file_type))

    return items


def get_embedded_items(client: QdrantClient, item_type: str = "all") -> List[Dict]:
    items = []
    seen_ids = set()
    
    results = client.scroll(
        collection_name="files",
        with_payload=True,
        with_vectors=False,
        limit=1000
    )
    
    if results[0]:
        for point in results[0]:
            payload = point.payload
            if payload and "path" in payload:
                item_id = payload["id"]
                is_file = "size" in payload
                
                if item_id not in seen_ids and (
                    (item_type == "all") or
                    (item_type == "file" and is_file) or
                    (item_type == "dir" and not is_file)
                ):
                    seen_ids.add(item_id)
                    items.append(payload)
    
    return items


def compare_items(current_items: List[Dict], embedded_items: List[Dict], item_type: str = "file") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    items_rm = []
    items_add = []
    items_mv = []
    
    embedded_items_dict = {item["id"]: item for item in embedded_items}
    
    for current_item in current_items:
        embedded_item = embedded_items_dict.get(current_item["id"])
        if embedded_item:
            if current_item["path"] != embedded_item["path"]:
                if item_type == "file" and current_item["modified"] == embedded_item["modified"]:
                    items_mv.append(current_item)
                elif item_type == "dir":
                    items_mv.append(current_item)
            elif item_type == "file" and current_item["modified"] != embedded_item["modified"]:
                    items_add.append(current_item)
                    items_rm.append(embedded_item)
        else:
            items_add.append(current_item)

    for embedded_item in embedded_items:
        if embedded_item["id"] not in {file["id"] for file in current_items}:
                    items_rm.append(embedded_item)
    
    return items_rm, items_add, items_mv


def remove_items(client: QdrantClient, items_rm: List[Dict], item_type: str = "file"):
    for item in items_rm:
        inode_id = item["id"]
        search_result = id_search(client, inode_id)
        
        if search_result:
            point_ids = [point.id for point in search_result]
            client.delete(
                collection_name="files",
                points_selector=models.PointIdsList(points=point_ids)
            )
            print(f"Removed {item_type} with path {item["path"]} from the vector database.")
        else:
            print(f"No {item_type} found with inode {inode_id} in the vector database.")


def move_items(client: QdrantClient, items_mv: List[Dict], model: AutoModel = None, item_type: str = "file"):
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps")

    for item in items_mv:
        inode_id = item["id"]
        new_name = item["name"]
        new_path = item["path"]

        search_result = id_search(client, inode_id, with_vectors=True)

        if search_result:
            updated_points = []
            for point in search_result:
                updated_payload = point.payload.copy()
                updated_payload["name"] = new_name
                updated_payload["path"] = new_path

                if updated_payload["type"] == f"{item_type}_name":
                    new_vector = embed_string(new_name, model)
                elif updated_payload["type"] == f"{item_type}_path":
                    new_vector = embed_string(new_path, model)
                else:
                    new_vector = point.vector

                updated_points.append(models.PointStruct(
                    id=point.id,
                    vector=new_vector,
                    payload=updated_payload
                ))

            client.upsert(
                collection_name="files",
                points=updated_points
            )
            print(f"Updated {len(updated_points)} points for {item_type} with inode {inode_id} in the vector database.")
        else:
            print(f"{item_type.capitalize()} with inode {inode_id} not found in the vector database.")


def copy_files(client: QdrantClient, files_cp: List[Dict], model: AutoModel = None):
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps") 
    for file_cp in files_cp:
        og_id = file_cp["og_id"]
        
        search_result = id_search(client, og_id, with_vectors=True)

        new_points = []
        for point in search_result:
            payload = {k: v for k, v in file_cp.items() if k != "og_id"}

            if point.payload["type"] == "file_name":
                new_vector = embed_string(file_cp["name"])
                payload["type"] = "file_name"
            elif point.payload["type"] == "file_path":
                new_vector = embed_string(file_cp["path"])
                payload["type"] = "file_path"
            else:
                new_vector = point.vector

            new_points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=new_vector,
                payload=payload
            ))

        if new_points:
            client.upsert(
                collection_name="files",
                points=new_points
            )
            print(f"Added {len(new_points)} new points for file with new ID {file_cp['id']} (original ID: {og_id}) in the vector database.")
        else:
            print(f"File with original ID {og_id} not found in the vector database.")


def update(client: QdrantClient, dirs: List[str], model: AutoModel = None):
    current_files = process_filesystem(dirs, include_dirs=False)
    current_dirs = process_filesystem(dirs, include_files=False)

    embedded_files = get_embedded_items(client, item_type="file")
    embedded_dirs = get_embedded_items(client, item_type="dir")

    files_rm, files_add, files_mv = compare_items(current_files, embedded_files, item_type="file")
    dirs_rm, dirs_add, dirs_mv = compare_items(current_dirs, embedded_dirs, item_type="dir")

    if files_mv or files_add or dirs_mv or dirs_add:
        if not model:
            model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps")

    if files_rm:
        remove_items(client, files_rm, item_type="file")
    if dirs_rm:
        remove_items(client, dirs_rm, item_type="directory")

    if files_mv:
        move_items(client, files_mv,  model=model, item_type="file")
    if dirs_mv:
        move_items(client, dirs_mv, model=model, item_type="directory")

    if files_add or dirs_add:
        add_all(client, files_add, dirs_add, model)