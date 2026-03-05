import os
from collections import defaultdict, Counter

def __init__(self):
    # term -> {doc_id: term_frequency}
    self.index = defaultdict(dict)

    # doc_id -> total number of terms (after preprocessing)
    self.document_lengths = {}

    # term -> document frequency
    self.document_frequency = defaultdict(int)

    # total number of documents
    self.N = 0

def index_collection(self, folder_path):

        # Reads all documents in the given folder and builds the inverted index.


        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            # Skip non-files
            if not os.path.isfile(file_path):
                continue

            doc_id = filename  # filename = document ID

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except:
                continue

            tokens = self.preprocess(text)

            if len(tokens) == 0:
                continue

            self.N += 1
            self.document_lengths[doc_id] = len(tokens)

            term_counts = Counter(tokens)

            for term, freq in term_counts.items():
                self.index[term][doc_id] = freq

            # Update document frequency
            for term in term_counts.keys():
                self.document_frequency[term] += 1

        print("Indexing completed.")
        print(f"Total documents indexed: {self.N}")
        print(f"Vocabulary size: {len(self.index)}")

