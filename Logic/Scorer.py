import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        doc_ids = set()
        for term in query:
            postings = self.index.get(term)
            if postings:
                doc_ids.update(postings)
        return list(doc_ids)

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term in self.idf:
            return self.idf[term]

        df = len(self.index.get(term, {}))
        if not df:
            return 0.0

        value = math.log10(self.N / df)
        self.idf[term] = value
        return value

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        tfs = {}
        for term in query:
            if term in tfs:
                tfs[term] += 1
            else:
                tfs[term] = 1
        return tfs
    
    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        doc_method, query_method = method.split('.')
        query_tfs = self.get_query_tfs(query)

        candidate_docs = self.get_list_of_documents(query)
        results = {}

        for doc_id in candidate_docs:
            score = self.get_vector_space_model_score(
                query, query_tfs, doc_id, doc_method, query_method
            )

            if score > 0:
                results[doc_id] = score

        return results

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        doc_weights = []
        query_weights = []

        for term in query_tfs:
            postings = self.index.get(term)
            if not postings:
                continue

            q_tf = self._apply_tf(query_tfs[term], query_method[0])
            q_idf = self.get_idf(term) if query_method[1] == "t" else 1.0
            query_weights.append(q_tf * q_idf)

            d_tf_raw = postings.get(document_id)
            if d_tf_raw is None:
                doc_weights.append(0.0)
                continue

            d_tf = self._apply_tf(d_tf_raw, document_method[0])
            d_idf = self.get_idf(term) if document_method[1] == "t" else 1.0
            doc_weights.append(d_tf * d_idf)

        if document_method[2] == 'c':
            doc_weights = self._cosine_normalize(doc_weights)
        if query_method[2] == 'c':
            query_weights = self._cosine_normalize(query_weights)

        return sum(d * q for d, q in zip(doc_weights, query_weights))

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        candidate_docs = self.get_list_of_documents(query)
        scores = {}
        
        for doc_id in candidate_docs:
            scores[doc_id] = self.get_okapi_bm25_score(
                query, doc_id, average_document_field_length, document_lengths
            )
            
        return scores

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        k1 = 1.5
        b = 0.75
        score = 0.0
        query_terms = set(query)
        doc_len = document_lengths.get(document_id, average_document_field_length)

        for term in query_terms:
            if term not in self.index:
                continue
            
            idf = self.get_idf(term)
            freq = self.index[term].get(document_id, 0)
            
            if freq > 0:
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / average_document_field_length))
                score += idf * (numerator / denominator)
        return score

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        candidate_docs = self.get_list_of_documents(query)
        scores = {}

        if smoothing_method != "laplace" and self._collection_frequencies is None:
            self._prepare_collection_stats()

        for doc_id in candidate_docs:
            scores[doc_id] = self.compute_score_with_unigram_model(
                query,
                doc_id,
                smoothing_method,
                document_lengths,
                alpha,
                lamda,
            )

        return scores

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        score = 0.0
        doc_len = document_lengths.get(document_id, 0)
        if doc_len == 0:
            return -1e9
        vocab_size = len(self.index)

        for term in query:
            tf_d = 0
            if term in self.index:
                tf_d = self.index[term].get(document_id, 0)

            if smoothing_method == 'laplace':
                p_t_d = (tf_d + 1) / (doc_len + vocab_size)
                score += math.log(p_t_d)

            elif smoothing_method == 'jm':
                if self._collection_frequencies is None:
                    self._prepare_collection_stats()
                
                p_td = tf_d / doc_len
                cf = self._collection_frequencies.get(term, 0)
                p_tc = cf / self._collection_length if self._collection_length > 0 else 1e-9
                prob = (lamda * p_td) + ((1 - lamda) * p_tc)
                score += math.log(prob) if prob > 0 else -1e9

            elif smoothing_method == 'dirichlet':
                if self._collection_frequencies is None:
                    self._prepare_collection_stats()
                
                cf = self._collection_frequencies.get(term, 0)
                p_tc = cf / self._collection_length if self._collection_length > 0 else 1e-9
                
                numerator = tf_d + (alpha * p_tc)
                denominator = doc_len + alpha
                score += math.log(numerator / denominator)
        return score

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if mode == 'n':
            return float(tf)
        elif mode == 'l':
            if tf > 0:
                return 1.0 + math.log10(tf)
            return 0.0
        return float(tf)

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        if not weights:
            return weights
        sq_sum = sum(w**2 for w in weights)
        norm = math.sqrt(sq_sum)
        
        if norm == 0:
            return weights
            
        return [w / norm for w in weights]

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        self._collection_frequencies = {}
        total_len = 0
        
        for term, postings in self.index.items():
            # Sum up all tfs for this term across all docs
            term_total = sum(postings.values())
            self._collection_frequencies[term] = term_total
            total_len += term_total
            
        self._collection_length = total_len
        return