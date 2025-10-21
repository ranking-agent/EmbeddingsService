from pydantic import BaseModel
from typing import Optional, List, Dict

class Relation(BaseModel):
    predicate: str
    object_aspect_qualifier: Optional[str] = None
    object_direction_qualifier: Optional[str] = None
    subject_aspect_qualifier: Optional[str] = None
    subject_direction_qualifier: Optional[str] = None

class Query(BaseModel):
    curies: Optional[List] = None
    node_embedding: Optional[List[float]] = None
    curie_embedding: Optional[Dict[str, List]] = None
    distance_threshold: Optional[float] = None
    curie_limit: Optional[int] = None
    relation: Optional[Relation] = None

class Response(BaseModel):
    curies: Optional[List] = None
    node_embedding: Optional[Dict[str, List]] = None
    logs: Optional[List] = None
    distance: Optional[float] = None