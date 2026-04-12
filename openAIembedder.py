#!/usr/bin/env python3
# What it does:
#  - Connects to Atlas using your connection string (below)
#  - Cleans HTML from Definition/Notes fields (TitleCase or camelCase)
#  - Builds an embedding text from key fields
#  - Calls OpenAI to create embeddings (text-embedding-3-large)
#  - Writes the vector to "embedding" and saves cleaned Definition/Notes
#
# NOTE: This script contains plaintext credentials because it's for quick demos.
#       Rotate/replace your API key after sharing/using in public contexts.

from pymongo import MongoClient, UpdateOne
import requests
import re, html, sys

# ---------- Hardcoded demo creds (as requested) ----------
MONGODB_URI = ""
DB_NAME     = "NUCC"
COLL_NAME   = "taxonomy251"

OPENAI_API_KEY = ""
OPENAI_MODEL   = "text-embedding-3-large"
EMBED_DIM      = 2048           # keep this in sync with your Atlas Vector Search index
BATCH_SIZE     = 128

# ---------- Connect ----------
client = MongoClient(MONGODB_URI)
coll = client[DB_NAME][COLL_NAME]

# ---------- Helpers ----------
TAG_RE = re.compile(r"<[^>]+>")

def strip_markup(s: str) -> str:
    """Remove HTML tags/entities. Keeps <br> as spaces. Collapses whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"(?i)<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(s.split())

def get(doc, *keys):
    for k in keys:
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def build_embedding_text(doc: dict) -> str:
    # Build from multiple fields; strip HTML where it might appear
    parts = [
        get(doc, "displayName", "Display Name"),
        get(doc, "classification", "Classification"),
        get(doc, "specialization", "Specialization"),
        strip_markup(get(doc, "definition", "Definition")),
        get(doc, "grouping", "Grouping"),
        get(doc, "section", "Section"),
        get(doc, "code", "Code"),
        strip_markup(get(doc, "notes", "Notes")),
    ]
    return " ".join([p for p in parts if p])

def embed_texts(texts):
    if not texts:
        return []

    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": texts,
            "dimensions": EMBED_DIM,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]

    # sort by index to be safe
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]

def main():
    projection = {
        "_id": 1,
        "displayName": 1, "Display Name": 1,
        "classification": 1, "Classification": 1,
        "specialization": 1, "Specialization": 1,
        "definition": 1, "Definition": 1,
        "grouping": 1, "Grouping": 1,
        "section": 1, "Section": 1,
        "code": 1, "Code": 1,
        "notes": 1, "Notes": 1,
    }

    cur = coll.find({}, projection=projection, no_cursor_timeout=True)

    batch, texts, ops = [], [], []
    processed = 0

    try:
        for doc in cur:
            # Clean fields if present
            updates = {}
            for fld in ("definition", "Definition", "notes", "Notes"):
                raw = doc.get(fld)
                if isinstance(raw, str) and raw:
                    cleaned = strip_markup(raw)
                    if cleaned != raw:
                        updates[fld] = cleaned

            # Build text for embedding (from cleaned view)
            working = {**doc, **updates}
            text = build_embedding_text(working)

            batch.append((doc["_id"], updates))
            texts.append(text)

            if len(batch) == BATCH_SIZE:
                write_batch(batch, texts, ops)
                processed += len(batch)
                print(f"Processed {processed} docs...", file=sys.stderr)
                batch, texts = [], []

        if batch:
            write_batch(batch, texts, ops)
            processed += len(batch)

        if ops:
            coll.bulk_write(ops)
            ops.clear()

    finally:
        cur.close()

    print(f"Done. Processed {processed} docs.")

def write_batch(batch, texts, ops):
    vectors = embed_texts(texts)
    for (doc_id, updates), vec in zip(batch, vectors):
        set_doc = {"embedding": vec}
        if updates:
            set_doc.update(updates)
        ops.append(UpdateOne({"_id": doc_id}, {"$set": set_doc}))

    if len(ops) >= 1000:
        coll.bulk_write(ops)
        ops.clear()

if __name__ == "__main__":
    main()
