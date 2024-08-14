from qdrant_client import models
from PIL import Image
import uuid
from typing import List, Dict, Any
import re
from reader import read_pdf, read_doc, read_csv, read_pptx
from globals import client, embedding_model


def embed_image(file: Dict) -> Any:
    image = Image.open(file["path"])
    return embedding_model.encode_image(image)


def embed_string(string: str) -> Any:
    return embedding_model.encode_text(string)


def add_chunks(file_dict: Dict):
    with open(file_dict["path"], 'r') as file:
        text = file.read()
    
    chunks = chunk_document(preprocess_file(text))

    for chunk in chunks:
        payload = file_dict.copy()
        payload["content"] = chunk
        client.upsert(
            collection_name="files",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_string(chunk),
                    payload=payload
                )],
        )


def add_image(file: Dict):
    client.upsert(
        collection_name="files",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_image(file),
                payload=file
            )],
    )

def add_pdf(file: Dict):
    pages = read_pdf(file["path"])

    for page in pages:
        chunks = chunk_document(preprocess_file(page["text"]))

        for chunk in chunks:
            payload = file.copy()
            payload["content"] = chunk
            payload["page"] = page["page"]
            client.upsert(
                collection_name="files",
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embed_string(chunk),
                        payload=payload
                    )],
            )

    
def add_doc(file: Dict):
    text = read_doc(file["path"])

    chunks = chunk_document(preprocess_file(text))

    for chunk in chunks:
        payload = file.copy()
        payload["content"] = chunk
        client.upsert(
            collection_name="files",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_string(chunk),
                    payload=payload
                )],
        )


def add_pptx(file: Dict):
    slides = read_pptx(file["path"])

    for slide in slides:
        chunks = chunk_document(preprocess_file(slide["text"]))

        for chunk in chunks:
            payload = file.copy()
            payload["content"] = chunk
            payload["slide"] = slide["slide"]
            client.upsert(
                collection_name="files",
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embed_string(chunk),
                        payload=payload
                    )],
            )


def add_csv(file: Dict):
    text = read_csv(file["path"])

    chunks = chunk_document(preprocess_file(text))

    for chunk in chunks:
        payload = file.copy()
        payload["content"] = chunk
        client.upsert(
            collection_name="files",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_string(chunk),
                    payload=payload
                )],
        )


def add_dir_embeddings(dirs: List[Dict]):   
    points = []

    for dir in dirs:
        dir_name = dir.copy()
        dir_name["type"] = "dir_name"
        dir_path = dir.copy()
        dir_path["type"] = "dir_path"
        points.extend([
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(dir["name"]),
                payload=dir_name
            ),
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(dir["path"]),
                payload=dir_path
            )
        ])

    client.upsert(
        collection_name="files",
        points=points)
    

def add_file_embeddings(files: List[Dict]):   
    points = []
    max_file_size = 20_000_000  # 20MB
    for file in files:
        if file["size"] < max_file_size:
            type = file["type"]
            if type == "image":
                add_image(file)
            elif type == "text":
                add_chunks(file)
            elif type == "pdf":
                add_pdf(file)
            elif type == "doc":
                add_doc(file)   
            elif type == "ppt":
                add_pptx(file)
            elif type == "csv":
                add_csv(file)
        else:
            print(f"{file['path']} is too large and it's content won't be embedded.")

        file_name = file.copy()
        file_name["type"] = "file_name"
        file_path = file.copy()
        file_path["type"] = "file_path"
        points.extend([
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(file["name"]),
                payload=file_name
            ),
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(file["path"]),
                payload=file_path
            )
        ])

    client.upsert(
        collection_name="files",
        points=points
    )


def add_all(files: List[Dict], dirs: List[Dict]):
    print("Starting embedding")
    if files:
        add_file_embeddings(files)
        print("Added files")
    if dirs:
        add_dir_embeddings(dirs)
        print("Added dirs")
    print("Finished embedding")


def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            last_period = text.rfind('.', end - overlap, end)
            last_newline = text.rfind('\n', end - overlap, end)
            
            break_point = max(last_period, last_newline)
            if break_point != -1:
                end = break_point + 1 
        
        chunk = text[start:end].strip()
        chunks.append(chunk)
        
        start = max(end - overlap, start + 1)
    
    return chunks


def preprocess_file(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()
