from transformers import AutoModel
from qdrant_client import QdrantClient, models
from PIL import Image
import uuid
from typing import List, Dict, Any
import re
from reader import read_pdf, read_doc, read_csv, read_pptx


def embed_image(file: Dict, model: AutoModel) -> Any:
    content = Image.open(file["path"])
    return model.encode_image(content)


def embed_string(string: str, model: AutoModel = None) -> Any:
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps") 
    return model.encode_text(string)


def add_chunks(client: QdrantClient, file_dict: Dict, model: AutoModel):
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps")
    
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
                    vector=embed_string(chunk, model),
                    payload=payload
                )],
        )


def add_image(client: QdrantClient, file: Dict, model: AutoModel):
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps")    
     
    client.upsert(
        collection_name="files",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_image(file, model),
                payload=file
            )],
    )

def add_pdf(client: QdrantClient, file: Dict, model: AutoModel):
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
                        vector=embed_string(chunk, model),
                        payload=payload
                    )],
            )

    
def add_doc(client: QdrantClient, file: Dict, model: AutoModel):
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
                    vector=embed_string(chunk, model),
                    payload=payload
                )],
        )


def add_pptx(client: QdrantClient, file: Dict, model: AutoModel):
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
                        vector=embed_string(chunk, model),
                        payload=payload
                    )],
            )


def add_csv(client: QdrantClient, file: Dict, model: AutoModel):
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
                    vector=embed_string(chunk, model),
                    payload=payload
                )],
        )


def add_dir_embeddings(client: QdrantClient, dirs: List[Dict], model: AutoModel):   
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps")
    points = []

    for dir in dirs:
        dir_name = dir.copy()
        dir_name["type"] = "dir_name"
        dir_path = dir.copy()
        dir_path["type"] = "dir_path"
        points.extend([
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(dir["name"], model),
                payload=dir_name
            ),
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(dir["path"], model),
                payload=dir_path
            )
        ])

    client.upsert(
        collection_name="files",
        points=points)
    

def add_file_embeddings(client: QdrantClient, files: List[Dict], model: AutoModel):   
    if not model:
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to("mps") 
    points = []
    max_file_size = 20_000_000  # 20MB
    for file in files:
        if file["size"] < max_file_size:
            type = file["type"]
            if type == "image":
                add_image(client, file, model)
            elif type == "text":
                add_chunks(client, file, model)
            elif type == "pdf":
                add_pdf(client, file, model)
            elif type == "doc":
                add_doc(client, file, model)   
            elif type == "ppt":
                add_pptx(client, file, model)
            elif type == "csv":
                add_csv(client, file, model)
        else:
            print(f"{file["path"]} is too large and it's content won't be embedded.")

        file_name = file.copy()
        file_name["type"] = "file_name"
        file_path = file.copy()
        file_path["type"] = "file_path"
        points.extend([
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(file["name"], model),
                payload=file_name
            ),
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_string(file["path"], model),
                payload=file_path
            )
        ])

    client.upsert(
        collection_name="files",
        points=points
    )


def add_all(client: QdrantClient, files: List[Dict], dirs: List[Dict], model: AutoModel):
    print("Starting embedding")
    if files:
        add_file_embeddings(client, files, model)
        print("Added files")
    if dirs:
        add_dir_embeddings(client, dirs, model)
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
