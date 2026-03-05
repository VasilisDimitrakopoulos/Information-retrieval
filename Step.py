import os
import re
import math
from collections import defaultdict, Counter

folder_path = "C:\\Users\\Vasilis\\Documents\\Vasilis\\Programming\\Ανάκτηση Πληροφορίας 2025-2026\\collection\\docs"

# Number of documents
N = int(0)
document_lengths = {}
index = defaultdict(dict)
document_frequency = defaultdict(int)

def preprocess(text):
        # Lowercasing
        # Remove punctuation
        # Tokenization

        
        # Lowercasing
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # Tokenize
        tokens = text.split()

        return tokens

# All docs of collection
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Skip if not a file
    if not os.path.isfile(file_path):
        continue

    doc_id = filename

    # Open each file
    # Try and catch error cause some files missing or are destroyed
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            contents = f.read()
    except Exception as e:
        print(f"Error reading {doc_id}: {e}")
        continue

    # Preprocess of text in every file and save
    tokens = preprocess(contents)
    # Increment in the number of the documents
    N += 1
    document_lengths[doc_id] = len(tokens)
    term_counts = Counter(tokens)

    # index -> {doc_id: term_frequency}
    for term, freq in term_counts.items():
        index[term][doc_id] = freq

    # In how many docs appears every word
    for term in term_counts.keys():
        document_frequency[term] += 1

# print("Indexing completed.")
# print(f"Total documents indexed: {N}")
# print(f"Vocabulary size: {len(index)}")
# print(f"Index: {index}")
# print(dict(index))
# print(index["fibrosis"])
# print(term_counts)

def compute_idf():
    idf = {}
    for term, df in document_frequency.items():
        idf[term] = math.log(N / df)
    return idf

def compute_document_tfidf(idf):
    doc_vectors = defaultdict(dict)

    for term, postings in index.items():
        for doc_id, tf in postings.items():
            weight = tf * idf[term]
            doc_vectors[doc_id][term] = weight

    return doc_vectors

def compute_query_tfidf(query, idf):
    tokens = preprocess(query)
    term_counts = Counter(tokens)

    query_vector = {}

    for term, tf in term_counts.items():
        if term in idf:
            query_vector[term] = tf * idf[term]

    return query_vector

def cosine_similarity(vec1, vec2):
    dot_product = 0
    norm1 = 0
    norm2 = 0

    for term, weight in vec1.items():
        norm1 += weight ** 2
        if term in vec2:
            dot_product += weight * vec2[term]

    for weight in vec2.values():
        norm2 += weight ** 2

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

def rank_documents(query, doc_vectors, idf):
    query_vector = compute_query_tfidf(query, idf)

    scores = {}

    for doc_id, doc_vector in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        scores[doc_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked

# IDF
idf = compute_idf()
# Check the first 5 values if > 0 
#   rare terms -> big values
#   frequent terms -> small
# print("Sample IDF values:")
# for i, (term, value) in enumerate(idf.items()):
#     print(term, value)
#     if i == 5:
#         break

# TF-IDF
doc_vectors = compute_document_tfidf(idf)
# print("\nSample document vector:")
# sample_doc = list(doc_vectors.keys())[0]
# print(sample_doc)
# print(doc_vectors[sample_doc])

query1 = "How effective are inhalations of mucolytic agents in the treatment of CF patients"
query2 = "What is the role of aerosols in the treatment of lung disease in CF patients"

print(rank_documents(query1, doc_vectors, idf)[:3])
print(rank_documents(query2, doc_vectors, idf)[:3])

# print("\nTop 10 Results:")
# for doc_id, score in results[:10]:
#     print(doc_id, score)