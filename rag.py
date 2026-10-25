import glob
import os

CHUNK, OVERLAP = 40, 10
STEP = CHUNK - OVERLAP

def chunk_file(path):
    words = open(path, encoding="utf-8").read().split()
    chunks, start = [], 0
    while True:
        end = start + CHUNK
        chunks.append({"source": os.path.basename(path),
                       "text": " ".join(words[start:end])})
        if end >= len(words):
            break
        start += STEP
    return chunks

chunks = []
for path in sorted(glob.glob("data/policies/*.txt")):
    file_chunks = chunk_file(path)
    chunks.extend(file_chunks)
    print(os.path.basename(path), len(file_chunks))

print("total chunks:", len(chunks))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
chunk_matrix = vectorizer.fit_transform([c["text"] for c in chunks])

def retrieve(query, k=3):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, chunk_matrix)[0]
    top = scores.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "score": round(float(scores[i]), 3),
             "text": chunks[i]["text"]} for i in top]

for hit in retrieve("refund policy late delivery"):
    print(hit["source"], hit["score"])