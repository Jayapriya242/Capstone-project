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