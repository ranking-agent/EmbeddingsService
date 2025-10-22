import psycopg2

conn = psycopg2.connect(
    "user='postgres' host='0.0.0.0' password='postgres'"
)
with conn.cursor() as cur:

    cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
                
DROP TABLE IF EXISTS public.node_embeddings_real;
                
DROP TABLE IF EXISTS public.node_embeddings_im;
                
DROP TABLE IF EXISTS public.node_embeddings;

DROP TABLE IF EXISTS public.edge_embeddings;

CREATE TABLE IF NOT EXISTS public.node_embeddings_real
(
    curie character varying NOT NULL,
    node_id integer,
    embedding VECTOR(20),
    PRIMARY KEY (curie)
);
CREATE TABLE IF NOT EXISTS public.node_embeddings_im
(
    curie character varying NOT NULL,
    node_id integer,
    embedding VECTOR(20),
    PRIMARY KEY (curie)
);
CREATE TABLE IF NOT EXISTS public.node_embeddings
(
    curie character varying NOT NULL,
    node_id integer,
    embedding VECTOR(40),
    PRIMARY KEY (curie)
);
CREATE TABLE IF NOT EXISTS public.edge_embeddings
(
    predicate character varying NOT NULL,
    object_aspect_qualifier character varying,
    object_direction_qualifier character varying,
    subject_aspect_qualifier character varying,
    subject_direction_qualifier character varying,
    edge_id integer NOT NULL,
    embedding VECTOR(20),
    PRIMARY KEY (edge_id)
);

-- Supported distance functions are:
-- <-> - L2 distance
-- <#> - (negative) inner product
-- <=> - cosine distance
-- <+> - L1 distance
-- <~> - Hamming distance (binary vectors)
-- <%> - Jaccard distance (binary vectors)
    """)

    with open("edge_emb_rows_49000.csv", "r") as f:
        cur.copy_expert(
            "COPY edge_embeddings FROM STDIN WITH (FORMAT CSV, HEADER FALSE)",
            f
        )
    with open("node_emb_rows_49000.csv", "r") as f:
        cur.copy_expert(
            "COPY node_embeddings_real FROM STDIN WITH (FORMAT CSV, HEADER FALSE)",
            f
        )
    with open("node_emb_im_rows_49000.csv", "r") as f:
        cur.copy_expert(
            "COPY node_embeddings_im FROM STDIN WITH (FORMAT CSV, HEADER FALSE)",
            f
        )

    cur.execute("""
INSERT INTO node_embeddings
SELECT
    real_emb.curie,
    real_emb.node_id,
    real_emb.embedding || im_emb.embedding AS embedding
FROM
    node_embeddings_real real_emb
JOIN
    node_embeddings_im im_emb ON real_emb.curie = im_emb.curie;
    """)

    cur.execute("CREATE INDEX ON node_embeddings USING hnsw (embedding vector_l2_ops) WITH (m = 32, ef_construction = 128);")
    cur.execute("SET hnsw.ef_search = 24;")
    conn.commit()