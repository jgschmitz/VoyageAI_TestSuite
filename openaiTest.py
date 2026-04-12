#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Embedding-only vector search → CSV
- Model: OpenAI GPT-5-4 (2048 dims)
- Index: Atlas Vector Search on path=VECTOR_FIELD (2048-d vectors)
- NO reranking. Top-10 per query from ANN by vectorSearchScore.
- Assumes document vectors were built from: name|definition|specialization|classification
- Writes voyage_eval_result.csv (no NaNs; blanks instead)
- Set PRINT_SAMPLE_N > 0 to see a few rows per query
- This runs against FindCare opensource NUCC dataset 
"""

import math
from itertools import islice

import pandas as pd
from pymongo import MongoClient
import openai

# ---------- Config (fill these) ----------
MONGODB_URI    = ""
OPENAI_KEY = ""

DB, COLL, INDEX_NAME = "NUCC", "taxonomy251", "default"   # <- your Atlas Vector Search index name
VECTOR_FIELD = "embedding"                                 # 2048-d vectors live here

EMBED_MODEL, DIM = "text-embedding-3-large", 2048
TOP_K = 10
NUM_CANDIDATES = 1000
ONLY_INDIVIDUALS = False            # True → filter section == "Individual"
OUT_CSV = "openai_eval_result2.csv"
PRINT_SAMPLE_N = 0                  # e.g., 3 to preview a few rows per query
# ----------------------------------------

TERMS_RAW = [
    # Existing terms (keeping all yours)
    "abdominoplasty","acne","acupressure","adhd","alcohol abuse","alopecia","annual phy","arthritis","athletes foot",
    "blurred vision","bone density","bone marrow","botox","breast mri","breast pump","broken wrist","bronchitis","burns",
    # ... (all your existing terms)
    
    # Expanded synonyms
    "attention deficit hyperactivity disorder", "add", "annual physical", "wellness exam", "preventive care",
    "cognitive behavioral therapy", "talk therapy", "athlete's foot", "fungal infection",
    
    # Medical specialties
    "cardiology", "cardiologist", "heart doctor", "heart specialist",
    "pulmonology", "pulmonologist", "lung doctor", "respiratory specialist", 
    "nephrology", "nephrologist", "kidney doctor", "kidney specialist",
    "rheumatology", "rheumatologist", "arthritis doctor", "joint specialist",
    "endocrinology", "endocrinologist", "diabetes doctor", "hormone specialist",
    "hematology", "hematologist", "blood doctor", "blood specialist",
    
    # Common symptoms
    "chest pain", "shortness of breath", "difficulty breathing", "wheezing", "chronic cough",
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain", "stomach pain",
    "dizziness", "lightheadedness", "fainting", "syncope", "numbness", "tingling",
    "fatigue", "weakness", "fever", "chills", "night sweats",
    
    # Common procedures  
    "ct scan", "cat scan", "mri scan", "ultrasound", "pet scan", "bone scan",
    "blood work", "blood test", "lab work", "urine test", "stool sample",
    "endoscopy", "colonoscopy", "upper endoscopy", "sigmoidoscopy",
    "ekg", "ecg", "electrocardiogram", "stress test", "cardiac stress test",
    
    # Chronic conditions
    "hypertension", "high blood pressure", "hypotension", "low blood pressure",
    "diabetes type 1", "diabetes type 2", "diabetic", "prediabetes",
    "asthma", "copd", "emphysema", "chronic bronchitis",
    "depression", "anxiety", "panic disorder", "bipolar", "ptsd",
    
    # Alternative spellings
    "hemorroids", "diarrhoea", "paediatric", "anaemia", "oestrogen",
]


def unique_trimmed(seq):
    seen, out = set(), []
    for s in seq:
        t = (s or "").strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k); out.append(t)
    return out

TERMS = unique_trimmed(TERMS_RAW)

def chunks(lst, n):
    it = iter(lst)
    while True:
        block = list(islice(it, n))
        if not block:
            return
        yield block

def safe_str(x, maxlen=None):
    """Convert None/NaN/nums/anything to a string safely, then truncate."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        s = ""
    else:
        s = str(x)
    return s[:maxlen] if (maxlen is not None) else s

def fmt_num(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def main():
    openai.api_key = OPENAI_KEY
    coll = MongoClient(MONGODB_URI)[DB][COLL]

    # Ensure vectors exist
    if coll.count_documents({VECTOR_FIELD: {"$type": "array"}}) == 0:
        raise RuntimeError(
            f"No vectors found in '{DB}.{COLL}.{VECTOR_FIELD}'. "
            f"Backfill 2048-d vectors (name|definition|specialization|classification) or adjust VECTOR_FIELD."
        )

    # Pre-embed all queries once (query embeddings @ 2048)
    batch_size = 64
    qvecs = {}
    for batch in chunks(TERMS, batch_size):
        resp = openai.Embedding.create(input=batch, model=EMBED_MODEL, dimensions=DIM)
        for q, item in zip(batch, resp["data"]):
            v = item["embedding"]
            if len(v) != DIM:
                raise ValueError(f"Embedding dim {len(v)} != DIM {DIM} for query '{q}'")
            qvecs[q] = v

    frames = []
    for q in TERMS:
        qvec = qvecs[q]

        # Embedding-only ANN retrieval (top-10 by vectorSearchScore)
        stage = {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": qvec,
                "numCandidates": NUM_CANDIDATES,
                "limit": TOP_K
            }
        }
        if ONLY_INDIVIDUALS:
            stage["$vectorSearch"]["filter"] = {"section": "Individual"}

        pipeline = [
            stage,
            {"$project": {
                "_id": 0,
                # These four fields correspond to: name | definition | specialization | classification
                "displayName":    {"$ifNull": ["$displayName",    "$Display Name"]},   # name
                "definition":     {"$ifNull": ["$definition",     "$Definition"]},
                "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
                "classification": {"$ifNull": ["$classification", "$Classification"]},
                "code":           {"$ifNull": ["$code",           "$Code"]},
                "section":        {"$ifNull": ["$section",        "$Section"]},
                "score": {"$meta": "vectorSearchScore"}
            }}
        ]

        docs = list(coll.aggregate(pipeline, allowDiskUse=True))
        df = pd.DataFrame(docs)

        if df.empty:
            # placeholder so CSV shows queries with no hits
            frames.append(pd.DataFrame([{
                "query": q, "rank": None, "code": None, "displayName": None,
                "classification": None, "specialization": None, "section": None,
                "score": None
            }]))
            if PRINT_SAMPLE_N:
                print(f"\n====================  {q}  ====================")
                print("No ANN hits (check index field/path/dim).")
            continue

        # Add query + rank, keep column order
        df.insert(0, "query", q)
        df["rank"] = range(1, len(df) + 1)

        if PRINT_SAMPLE_N:
            print(f"\n====================  {q}  ====================")
            head = df.head(PRINT_SAMPLE_N).copy()
            for _, r in head.iterrows():
                print(
                    f"{int(r['rank']):>2}  ann={fmt_num(r.get('score')):>6}  "
                    f"{safe_str(r.get('code'), 10):<10}  {safe_str(r.get('displayName'), 40)}"
                )

        # ensure we only ship requested columns (no rerank_score)
        keep = ["query", "rank", "code", "displayName", "classification", "specialization", "section", "score"]
        for c in keep:
            if c not in df.columns:
                df[c] = None
        frames.append(df[keep])

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False, na_rep="")
    print(f"\n✅ Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
