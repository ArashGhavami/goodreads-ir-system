import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        # self.hash_params = [
        # (random.randint(1, 10**6), random.randint(0, 10**6))
        # for _ in range(num_hashes)]

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(document) - k + 1):
            shingles.add(document[i:i + k])
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingles = [self.shingle_document(doc) for doc in self.documents]
        all_shingles_set = set()
        for shingles in doc_shingles:
            all_shingles_set = all_shingles_set.union(shingles)
        all_shingles = sorted(all_shingles_set) 
        index = {s: i for i, s in enumerate(all_shingles)}

        matrix = np.zeros((len(all_shingles), len(self.documents)), dtype=int)
        for j, sh_set in enumerate(doc_shingles):
            for s in sh_set:
                matrix[index[s], j] = 1
        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        mat = self.build_characteristic_matrix()
        num_rows, num_cols = mat.shape
        sig = np.full((self.num_hashes, num_cols), np.inf)

        for i in range(self.num_hashes):
            perm = np.random.permutation(num_rows)
            for pos, row_idx in enumerate(perm):
                cols_with_one = (mat[row_idx] == 1) & (sig[i] == np.inf)
                sig[i, cols_with_one] = pos
                if np.all(sig[i] != np.inf):
                    break

        return sig

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        num_docs = signature.shape[1]
        final_end = signature.shape[0]

        for b in range(bands):
            start = b * rows_per_band
            end = min(start + rows_per_band, final_end)

            for doc_id in range(num_docs):
                chunk = tuple(signature[start:end, doc_id])
                key = hash((b, chunk))

                if key not in buckets:
                    buckets[key] = []

                if doc_id not in buckets[key]:
                    buckets[key].append(doc_id)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes//num_bands)
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        if len(union) == 0:            
            return 0.0         
        return len(intersection) / len(union)

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

def main():
    file_path = "LSHFakeData.json"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("couldn't find the json file")
        return
    except json.JSONDecodeError:
        print("json file is broken or not valid")
        return

    docs = []
    for x in data:
        if "descriptions" in x and x["descriptions"]:
            docs.append(x["descriptions"][0])

    if len(docs) == 0:
        print("no documents loaded")
        return

    print(f"loaded {len(docs)} docs")
    num_hashes = 100
    print("starting lsh...")
    lsh = MinHashLSH(docs, num_hashes)
    buckets = lsh.perform_lsh()
    print("\nresults:")
    lsh.jaccard_similarity_test(buckets, docs)
    return

if __name__ == '__main__':
    main()