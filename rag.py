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
from mock_llm import chat

SYSTEM = ("POLICY ASSISTANT. Answer only from the supplied context. "
          "If the context does not contain the answer, say: I could not find that information in the policy documents.")

def answer(question):
    hits = retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

text, sources = answer("What delivery time makes a completed order eligible for a full refund?")
print("answer:", text)
print("sources:", sources)
import pandas as pd

ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
answerable = ev[ev["expected_doc"] != "NONE"]
unanswerable = ev[ev["expected_doc"] == "NONE"]

rows = []
for _, q in ev.iterrows():
    text, sources = answer(q["question"])
    rows.append({
        "question": q["question"],
        "expected_doc": q["expected_doc"],
        "doc_in_top3": q["expected_doc"] in sources,
        "fact_in_answer": q["expected_fact"] in text,
        "fallback": text.startswith("I could not find"),
    })
res = pd.DataFrame(rows)

ans_res = res.loc[answerable.index]
una_res = res.loc[unanswerable.index]
print("answerable: doc in top 3:", ans_res["doc_in_top3"].sum(), "of", len(ans_res))
print("answerable: fact in answer:", ans_res["fact_in_answer"].sum(), "of", len(ans_res))
print("unanswerable: fallback:", una_res["fallback"].sum(), "of", len(una_res))
print()
print(res[["doc_in_top3", "fact_in_answer", "fallback"]].assign(
    question=res["question"].str[:50]).to_string())

































