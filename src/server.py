from fastapi import FastAPI, Body, HTTPException
import httpx
from pydantic import HttpUrl, ValidationError

from src.operations import (
    get_embedding,
    find_neighbors,
    find_curies,
    get_distance_between,
    get_distance_from,
    find_node_embedding,
    find_node_embedding_from_embedding
)
from src.models import Query, Response

ONE_CURIE_EXAMPLE = {
    "curies": ["CHEBI:45783"],
    "relation": {
        "predicate": "biolink:treats",
        "object_aspect_qualifier": "None",
        "object_direction_qualifier": "None",
        "subject_aspect_qualifier": "None",
        "subject_direction_qualifier": "None"
    },
    "distance_threshold": 2,
    "curie_limit": 10,
    "node_embedding": [
        0.838866,
        0.8046492,
        -1.5814703,
        0.062232208,
        0.34491706,
        -0.5771061,
        -1.6293626,
        0.74601793,
        1.0437633,
        -1.3667237,
        0.6509505,
        0.50290596,
        -0.46790838,
        1.4808078,
        -0.08705791,
        0.5568928,
        0.38659558,
        0.38973445,
        -2.8049326,
        1.7076443,
        -0.9205438,
        1.5873375,
        -0.85251707,
        0.87959486,
        0.45769846,
        1.713986,
        1.0479169,
        1.483754,
        -0.6646749,
        -0.5344316,
        1.1289668,
        1.9131007,
        -0.21213739,
        0.82032347,
        -0.37374222,
        -0.32628623,
        0.2033696,
        0.28324828,
        2.3744113,
        -0.50940543
    ],
    "curie_embedding": {
        "CHEBI:45783": [
            0.81750953,
            1.0488763,
            -1.5758717,
            0.2237116,
            0.27225772,
            -0.8299847,
            -1.6104082,
            0.6143499,
            0.9770484,
            -1.2727942,
            0.57596195,
            0.7453773,
            -0.40145895,
            1.530515,
            -0.120044276,
            0.545806,
            0.73505294,
            0.42978016,
            -2.8879676,
            1.4999776,
            -0.8597314,
            1.692653,
            -0.961887,
            0.6565495,
            0.6609372,
            1.7206875,
            0.7943472,
            1.4054924,
            -0.6167662,
            -0.59227514,
            0.86384326,
            2.0739682,
            -0.23538697,
            0.58989894,
            -0.36690766,
            -0.42463586,
            0.13450661,
            0.2837754,
            2.0861297,
            -0.43254542
        ]
    }
}

TWO_CURIE_EXAMPLE = {
    "curies": ["CHEBI:45783", "CHEBI:49603"],
}

title = "Embeddings Service"
APP = FastAPI(title=title)


@APP.post(
    "/get_embedding",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def get_embedding_for_curie(query: Query = Body(..., example=ONE_CURIE_EXAMPLE)):
    if query.curies:
        embedding = get_embedding(query.curies)
        if embedding:
            response = Response.model_validate(
                {
                    "node_embedding": embedding
                }
            )
            return response


@APP.post(
    "/find_neighbors",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def find_neighbors_for_curie(query: Query = Body(..., example=ONE_CURIE_EXAMPLE)):
    if query.curies and len(query.curies) == 1:
        if query.curie_limit:
            neighbors = find_neighbors(query.curies[0], query.distance_threshold, query.curie_limit)
        else:
            neighbors = find_neighbors(query.curies[0], query.distance_threshold)
        response = Response.model_validate(
            {
                "curies": neighbors
            }
        )
        return response


@APP.post(
    "/find_curies",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def find_curies_for_embedding(query: Query = Body(..., example=ONE_CURIE_EXAMPLE)):
    embedding = query.node_embedding
    if embedding and len(embedding) == 40:
        if query.curie_limit:
            curies = find_curies(embedding, query.distance_threshold, query.curie_limit)
        else:
            curies = find_curies(embedding, query.distance_threshold)
        return Response.model_validate(
            {
                "curies": curies if curies else []
            }
        )


@APP.post(
    "/get_distance_between_curies",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def get_distance_between_curies(query: Query = Body(..., example=TWO_CURIE_EXAMPLE)):
    curies = query.curies
    if curies and len(curies) == 2:
        distance = get_distance_between(curies[0], curies[1])
        if distance is not None:
            return Response.model_validate(
                {
                    "distance": distance
                }
            )


@APP.post(
    "/get_distance_from_embedding",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def get_distance_from_embedding(query: Query = Body(..., example=ONE_CURIE_EXAMPLE)):
    curies = query.curies
    embedding = query.node_embedding
    if (
        curies
        and len(curies) == 1
        and embedding
        and len(embedding) == 40
    ):
        distance = get_distance_from(curies[0], embedding)
        if distance is not None:
            return Response.model_validate(
                {
                    "distance": distance
                }
            )


@APP.post(
    "/find_node_embedding",
    response_model=Response,
    response_model_exclude_unset=True,
    responses={
        200: {
            "content": {"application/json": {"example": ""}},
        },
    },
)
def predict_node_from_relation(query: Query = Body(..., example=ONE_CURIE_EXAMPLE)):
    relation = query.relation
    if (
        query.curies
        and relation
    ):
        node_embedding = find_node_embedding(
            query.curies,
            relation.predicate,
            relation.object_aspect_qualifier,
            relation.object_direction_qualifier,
            relation.subject_aspect_qualifier,
            relation.subject_direction_qualifier
        )
        if node_embedding is not None:
            return Response.model_validate(
                {
                    "node_embedding": node_embedding
                }
            )
        else:
            return Response.model_validate(
                {}
            )
    elif (
        query.curie_embedding
        and relation
    ):
        node_embedding = find_node_embedding_from_embedding(
            query.curie_embedding,
            relation.predicate,
            relation.object_aspect_qualifier,
            relation.object_direction_qualifier,
            relation.subject_aspect_qualifier,
            relation.subject_direction_qualifier
        )
        if node_embedding is not None:
            return Response.model_validate(
                {
                    "node_embedding": node_embedding
                }
            )
        else:
            return Response.model_validate(
                {}
            )
    else:
        return Response.model_validate(
                {}
            )
    