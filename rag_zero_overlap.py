import glob
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat

CHUNK = 40
SYSTEM = ("POLICY ASSISTANT. Answer only from the supplied context. "
          "If the context does not contain the answer, say: I could not find that information in the policy documents.")

def chunk_file(path, overlap):
    words = open(path, encoding="utf-8").read().split()
    step = CHUNK - overlap
    chunks, start = [], 0
    while True:
        end = start + CHUNK
        chunks.append({"source": os.path.basename(path),
                       "text": " ".join(words[start:end])})
        if end >= len(words):
            break
        start += step
    return chunks

def build(overlap):
    chunks = []
    for path in sorted(glob.glob("data/policies/*.txt")):
        chunks.extend(chunk_file(path, overlap))
    vec = TfidfVectorizer()
    matrix = vec.fit_transform([c["text"] for c in chunks])
    return chunks, vec, matrix

def retrieve(query, chunks, vec, matrix, k=3):
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    top = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top]

def answer(question, chunks, vec, matrix):
    hits = retrieve(question, chunks, vec, matrix)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
chunks, vec, matrix = build(overlap=0)
print("total chunks with zero overlap:", len(chunks))

rows = []
for _, q in ev.iterrows():
    text, sources = answer(q["question"], chunks, vec, matrix)
    rows.append({
        "question": q["question"],
        "expected_doc": q["expected_doc"],
        "doc_in_top3": q["expected_doc"] in sources,
        "fact_in_answer": q["expected_fact"] in text,
        "answer": text,
    })
res = pd.DataFrame(rows)

ans = res[ev["expected_doc"].values != "NONE"]
una = res[ev["expected_doc"].values == "NONE"]
print("answerable: doc in top 3:", ans["doc_in_top3"].sum(), "of", len(ans))
print("answerable: fact in answer:", ans["fact_in_answer"].sum(), "of", len(ans))
print("unanswerable: fallback:", una["answer"].str.startswith("I could not find").sum(), "of", len(una))

print()
broken = ans[~ans["fact_in_answer"]]
for _, r in broken.iterrows():
    print("BROKEN:", r["question"])
    print("  expected:", r["expected_doc"], "| answer:", r["answer"])