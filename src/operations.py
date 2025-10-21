from collections import defaultdict
import psycopg2
import numpy
import json


def get_embedding(curie):
    conn = psycopg2.connect(
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
    )
    with conn.cursor() as cur:
        query = """SELECT curie, embedding
        FROM node_embeddings
        WHERE curie IN %s;"""

        cur.execute(query, (tuple(curie),))
        resp = cur.fetchall()
        curie_embedding = {}
        if resp:
            for res in resp:
                curie_embedding[res[0]] = json.loads(res[1])
        else:
            return None
        return curie_embedding


def find_neighbors(curie, distance_threshold=None, num_neighbors=5):
    conn = psycopg2.connect(
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
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
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
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
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
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
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
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
    curies,
    predicate,
    object_aspect_qualifier="None",
    object_direction_qualifier="None",
    subject_aspect_qualifier="None",
    subject_direction_qualifier="None"
):
    conn = psycopg2.connect(
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
    )

    with conn.cursor() as cur:
        query = """SELECT curie, embedding
        FROM node_embeddings_real
        WHERE curie IN %s;"""

        cur.execute(query, (tuple(curies),))
        curie_embedding = defaultdict(dict)
        for curie_real in cur.fetchall():
            if curie_real:
                curie_embedding[curie_real[0]]["real"] = json.loads(curie_real[1])
            else:
                return {}
        
        query = """SELECT curie, embedding
        FROM node_embeddings_im
        WHERE curie IN %s;"""

        cur.execute(query, (tuple(curies),))
        for curie_im in cur.fetchall():
            if curie_im:
                curie_embedding[curie_im[0]]["im"] = json.loads(curie_im[1])
            else:
                return {}
        
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
            return {}
        pred_real = numpy.cos(pred_emb)
        pred_im = numpy.sin(pred_emb)
        node_emb = {}
        for curie, curie_emb in curie_embedding.items():
            node_emb[curie] = [
                float(emb_val)
                for emb_val in curie_emb["real"] * pred_real - curie_emb["im"] * pred_im
            ]
            node_emb[curie].extend(
                [
                    float(emb_val)
                    for emb_val in curie_emb["real"] * pred_im + curie_emb["im"] * pred_real
                ]
            )
        
        return node_emb
    
def find_node_embedding_from_embedding(
    curie_embedding_mapping,
    predicate,
    object_aspect_qualifier="None",
    object_direction_qualifier="None",
    subject_aspect_qualifier="None",
    subject_direction_qualifier="None"
):
    conn = psycopg2.connect(
        "user='postgres' host='0.0.0.0' password='postgres' port='5432'"
    )

    with conn.cursor() as cur:

        curie_embedding = defaultdict(dict)
        for curie, emb in curie_embedding_mapping.items():
            curie_embedding[curie]["re"] = emb[:20]
            curie_embedding[curie]["im"] = emb[20:]
        
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
            return {}
        pred_re = numpy.cos(pred_emb)
        pred_im = numpy.sin(pred_emb)
        node_emb = {}
        for curie, curie_emb in curie_embedding.items():
            node_emb[curie] = [
                float(emb_val)
                for emb_val in curie_emb["re"] * pred_re - curie_emb["im"] * pred_im
            ]
            node_emb[curie].extend(
                [
                    float(emb_val)
                    for emb_val in curie_emb["re"] * pred_im + curie_emb["im"] * pred_re
                ]
            )
        
        return node_emb