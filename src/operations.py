import psycopg2
import numpy
import json


def get_embedding(curie) -> str | None:
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )
    with conn.cursor() as cur:
        query = """SELECT embedding
        FROM node_embeddings
        WHERE curie = %s;"""

        cur.execute(query, (curie,))
        res = cur.fetchone()
        if res:
            return json.loads(res[0])
        else:
            return None


def find_neighbors(curie, distance_threshold=None, num_neighbors=5):
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )

    with conn.cursor() as cur:
        if distance_threshold:
            query = """SELECT curie
            FROM node_embeddings
            WHERE curie != %s AND embedding <->
            (SELECT embedding FROM node_embeddings WHERE curie = %s) < %s
            ORDER BY embedding
            LIMIT %s;"""
            values = (curie, curie, distance_threshold, num_neighbors)
        else:
            query = """SELECT curie
            FROM node_embeddings
            WHERE curie != %s
            ORDER BY embedding <->
            (SELECT embedding FROM node_embeddings WHERE curie = %s)
            LIMIT %s;"""
            values = (curie, curie, num_neighbors)

        cur.execute(query, values)
        return [res[0] for res in cur.fetchall()]


def find_curies(embedding_list, distance_threshold=None, num_curies=5):
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )

    embedding = f"[{','.join(str(emb) for emb in embedding_list)}]"

    with conn.cursor() as cur:
        if distance_threshold:
            query = """SELECT curie
            FROM node_embeddings
            WHERE embedding <-> %s < %s
            ORDER BY embedding
            LIMIT %s;"""
            values = (embedding, distance_threshold, num_curies)
        else:
            query = """SELECT curie
            FROM node_embeddings
            ORDER BY embedding <-> %s
            LIMIT %s;"""
            values = (embedding, num_curies)

        cur.execute(query, values)
        return [res[0] for res in cur.fetchall()]


def get_distance_between(curie1, curie2):
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )

    with conn.cursor() as cur:
        query = """SELECT
        (SELECT embedding FROM node_embeddings WHERE curie = %s) <->
        (SELECT embedding FROM node_embeddings WHERE curie = %s)
        AS distance
        FROM node_embeddings;"""
        cur.execute(query, (curie1, curie2))
        res = cur.fetchone()
        if res:
            return res[0]
        else:
            return None


def get_distance_from(curie, embedding_list):
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )

    embedding = f"[{','.join(str(emb) for emb in embedding_list)}]"

    with conn.cursor() as cur:
        query = """SELECT
        (SELECT embedding FROM node_embeddings WHERE curie = %s) <-> %s
        AS distance
        FROM node_embeddings;"""
        cur.execute(query, (curie, embedding))
        res = cur.fetchone()
        if res:
            return res[0]
        else:
            return None


def find_node_embedding(
    curie,
    predicate,
    object_aspect_qualifier,
    object_direction_qualifier,
    subject_aspect_qualifier,
    subject_direction_qualifier
):
    conn = psycopg2.connect(
        "user='postgres' host='localhost' password='postgres' port='5432'"
    )

    with conn.cursor() as cur:
        query = """SELECT embedding
        FROM node_embeddings_real
        WHERE curie = %s;"""

        cur.execute(query, (curie,))
        curie_real = cur.fetchone()
        if curie_real:
            curie_real = json.loads(curie_real[0])
        else:
            return
        
        query = """SELECT embedding
        FROM node_embeddings_im
        WHERE curie = %s;"""

        cur.execute(query, (curie,))
        curie_im = cur.fetchone()
        if curie_im:
            curie_im = json.loads(curie_im[0])
        else:
            return
        
        query = """SELECT embedding
        FROM edge_embeddings
        WHERE predicate = %s
        AND object_aspect_qualifier = %s
        AND object_direction_qualifier = %s
        AND subject_aspect_qualifier = %s
        AND subject_direction_qualifier = %s;"""

        cur.execute(
            query,
            (
                predicate,
                object_aspect_qualifier,
                object_direction_qualifier,
                subject_aspect_qualifier,
                subject_direction_qualifier
            )
        )
        pred_emb = cur.fetchone()
        if pred_emb:
            pred_emb = json.loads(pred_emb[0])
        else:
            return
        pred_real = numpy.cos(pred_emb)
        pred_im = numpy.sin(pred_emb)
        node_real = curie_real * pred_real - curie_im * pred_im
        node_im = curie_real * pred_im + curie_im * pred_real
        node_emb = [
            float(val)
            for val in node_real
        ]
        node_emb.extend(
            [
                float(val)
                for val in node_im
            ]
        )
        return node_emb
    
