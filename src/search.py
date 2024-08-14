from qdrant_client import models
from qdrant_client.conversions.common_types import ScoredPoint
from embed import embed_string
from typing import List
from globals import client


def vec_search(query: str, limit: int = 1): 
    result = client.search(
        collection_name="files",
        query_vector=embed_string(query),
        limit=limit)

    return result[0]


def id_search(inode_id: str, with_vectors=False) -> List[ScoredPoint]:
    search_result = client.search(
        collection_name="files",
        query_vector=[0.0] * 768,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=inode_id)
                )
            ]
        ),
        limit=1000,
        with_vectors=with_vectors
        
    )
    
    return search_result


def rag_search(query: str, limit: int = 3):
    result = client.search(
        collection_name="files",
        query_vector=embed_string(query),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchAny(any=["text", "doc", "csv", "pdf", "ppt"])
                )
            ]
        ),
        limit=limit
    )
    
    return result

