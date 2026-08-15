import json                  # chunk metadata is stored as JSON strings
import os                    # reads the API keys out of the environment
import pickle                # saves the BM25 sparse index to a file
import re                    # splits article text on paragraph breaks
import uuid                  # gives every chunk a unique id
from typing import Any

import chromadb                                 # the local vector database
from chromadb.config import Settings            # disables the telemetry that errors on this posthog version
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,       # wraps the embedding model
)
from deltalake import write_deltalake           # writes a Delta table, no Spark needed
import pandas as pd                             # reads plan_terms.csv
from dotenv import load_dotenv                  # loads .env into the environment
import ollama                                   # talks to the local fallback model

# Groq's client, plus the three errors the fallback in 4.3 catches by name.
from groq import Groq, APIError as GroqAPIError, AuthenticationError, RateLimitError

load_dotenv()                # must run before anything reads os.environ

# Where each store lives. All three paths are relative to the project root,
# so run the script from there.
CHROMA_DIR = "./chroma_store"             # dense index, the embeddings
COLLECTION_NAME = "support_articles"      # the collection name inside Chroma
SPARSE_INDEX_PATH = "./sparse_index.pkl"  # sparse index, the BM25 keyword half
DELTA_TABLE_PATH = "./delta_plan_terms"   # the structured plan terms table

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # runs locally, downloaded on first use
GROQ_MODEL = "llama-3.3-70b-versatile"      # primary generator, hosted and fast
OLLAMA_MODEL = "llama3.2"                   # fallback generator, local and free

CHUNK_SIZE = 700             # characters per chunk, before context is added
CHUNK_OVERLAP = 100          # characters repeated between neighbouring chunks

CONTEXT_PROMPT_TEMPLATE = """Here is a full support article, followed by one \
chunk taken from it. Write a single sentence of context, no more than 30 \
words, that states which plan and which topic this chunk is about, so the \
chunk makes sense even when read completely on its own, without the rest \
of the article.

Full article:
{full_text}

Chunk to contextualize:
{chunk}

Respond with only the one-sentence context, nothing else."""


def load_plan_terms_to_delta(csv_path: str, delta_path: str = DELTA_TABLE_PATH) -> None:
    df = pd.read_csv(csv_path)
    write_deltalake(delta_path, df, mode="overwrite")
    print(f"Loaded {len(df)} plan term rows into Delta table at {delta_path}")


def call_llm_with_fallback(messages: list[Any]) -> str | None:
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
        return response.choices[0].message.content
    except (AuthenticationError, RateLimitError) as e:
        print(f"  [fallback] Groq unavailable ({type(e).__name__}), using local Ollama...")
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]
    except GroqAPIError as e:
        print(f"  [fallback] Groq API error ({e}), using local Ollama...")
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]


def generate_context(full_text: str, chunk: str) -> str:
    prompt = CONTEXT_PROMPT_TEMPLATE.format(full_text=full_text, chunk=chunk)
    context = call_llm_with_fallback([{"role": "user", "content": prompt}])
    return context.strip() if context else ""


def load_documents(folder: str) -> list[dict]:
    documents = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            documents.append({"source": filename, "text": f.read()})
    return documents


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        flush()
        if len(para) <= chunk_size:
            current = para
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sentence in sentences:
            candidate = f"{buf} {sentence}".strip()
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = sentence[:chunk_size] if len(sentence) > chunk_size else sentence
        if buf:
            current = buf
    flush()

    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0 or overlap <= 0:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunk}")
    return overlapped


def build_hybrid_index(folder: str) -> None:
    documents = load_documents(folder)
    if not documents:
        print(f"No .txt files found in {folder}.")
        return

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn  # type: ignore[arg-type]
    )

    all_texts, all_ids, all_metadatas = [], [], []

    for doc in documents:
        raw_chunks = recursive_split(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"{doc['source']}: {len(raw_chunks)} chunks, generating context...")
        for i, chunk in enumerate(raw_chunks):
            context = generate_context(doc["text"], chunk)
            contextualized = f"{context}\n\n{chunk}"

            all_texts.append(contextualized)
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append({
                "source": doc["source"],
                "chunk_index": i,
                "raw_chunk": chunk,
                "generated_context": context,
            })

    # Dense index: Chroma embeds and stores the contextualized text.
    collection.add(documents=all_texts, ids=all_ids, metadatas=all_metadatas)

    # Sparse index: save the same contextualized texts + metadata for BM25,
    # rebuilt at query time in retrieve.py (Chapter 5).
    with open(SPARSE_INDEX_PATH, "wb") as f:
        pickle.dump({"texts": all_texts, "metadatas": all_metadatas}, f)

    print(f"\nIngested {len(documents)} article(s) into {len(all_texts)} contextualized chunks.")
    print(f"Dense index: {CHROMA_DIR}")
    print(f"Sparse index source: {SPARSE_INDEX_PATH}")

if __name__ == "__main__":
    load_plan_terms_to_delta("data/plan_terms.csv")
    build_hybrid_index("data/support_articles")
