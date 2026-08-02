import time
import os
import sys
import json
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Logic.indexer.indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.CHARACTERS.value: self.index_characters(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.DESCRIPTIONS.value: self.index_descriptions(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_characters(self):
        """
        Index the documents based on the characters.

        Returns
        ----------
        dict
            The index of the documents based on the characters. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            characters = doc.get('characters', [])
            if isinstance(characters, str):
                characters = characters.split()
            for term in characters:
                term = term.lower().strip()
                if not term:
                    continue
                if term not in index:
                    index[term] = {}
                index[term][doc_id] = index[term].get(doc_id, 0) + 1
        return index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            genres = doc.get('genres', [])
            if isinstance(genres, str):
                genres = genres.split()
            for term in genres:
                term = term.lower().strip()
                if not term:
                    continue
                if term not in index:
                    index[term] = {}
                index[term][doc_id] = index[term].get(doc_id, 0) + 1
        return index

    def index_descriptions(self):
        """
        Index the documents based on the descriptions.

        Returns
        ----------
        dict
            The index of the documents based on the descriptions. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            description = doc.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)
            if not description:
                continue
            for term in description.lower().split():
                term = term.strip()
                if not term:
                    continue
                if term not in current_index:
                    current_index[term] = {}
                current_index[term][doc_id] = current_index[term].get(doc_id, 0) + 1
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, characters, genres, descriptions)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return list(self.index[index_type][word].keys())
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        doc_id = document['id']

        self.index[Indexes.DOCUMENTS.value][doc_id] = document
        for term in document.get('characters', []):
            term = term.lower().strip()
            if not term:
                continue
            if term not in self.index[Indexes.CHARACTERS.value]:
                self.index[Indexes.CHARACTERS.value][term] = {}
            self.index[Indexes.CHARACTERS.value][term][doc_id] = \
                self.index[Indexes.CHARACTERS.value][term].get(doc_id, 0) + 1
        for term in document.get('genres', []):
            term = term.lower().strip()
            if not term:
                continue
            if term not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][term] = {}
            self.index[Indexes.GENRES.value][term][doc_id] = \
                self.index[Indexes.GENRES.value][term].get(doc_id, 0) + 1
        description = document.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description)
        for term in description.lower().split():
            term = term.strip()
            if not term:
                continue
            if term not in self.index[Indexes.DESCRIPTIONS.value]:
                self.index[Indexes.DESCRIPTIONS.value][term] = {}
            self.index[Indexes.DESCRIPTIONS.value][term][doc_id] = \
                self.index[Indexes.DESCRIPTIONS.value][term].get(doc_id, 0) + 1

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        self.index[Indexes.DOCUMENTS.value].pop(document_id, None)

        for index_type in [Indexes.CHARACTERS.value, Indexes.GENRES.value, Indexes.DESCRIPTIONS.value]:
            terms_to_delete = []
            for term, postings in self.index[index_type].items():
                postings.pop(document_id, None)
                if not postings:
                    terms_to_delete.append(term)
            for term in terms_to_delete:
                del self.index[index_type][term]

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if not index_before_add[index].__contains__(key):
            index_before_add[index].setdefault(key, {})

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'characters': ['sandman', 'robin'],
            'genres': ['mystery', 'crime'],
            'description': 'good'
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return


        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'sandman')

        if (set(index_after_add[Indexes.CHARACTERS.value]['sandman']).difference(set(index_before_add[Indexes.CHARACTERS.value]['sandman']))
                != {dummy_document['id']}):
            print('Add is incorrect, sandman')
            return

        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'robin')

        if (set(index_after_add[Indexes.CHARACTERS.value]['robin']).difference(set(index_before_add[Indexes.CHARACTERS.value]['robin']))
                != {dummy_document['id']}):
            print('Add is incorrect, robin')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'mystery')

        if (set(index_after_add[Indexes.GENRES.value]['mystery']).difference(set(index_before_add[Indexes.GENRES.value]['mystery']))
                != {dummy_document['id']}):
            print('Add is incorrect, mystery')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'crime')

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        self.check_if_key_exists(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        if (set(index_after_add[Indexes.DESCRIPTIONS.value]['good']).difference(set(index_before_add[Indexes.DESCRIPTIONS.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'sandman')
        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'robin')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'mystery')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str = 'C:\\Users\\amir\\Desktop\\codes\\MIR-Project-SP2026\\answer\\indexes\\', index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, characters, genres, descriptions)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        file_path = os.path.join(path, f"{index_name}_index.json")
        with open(file_path, 'w') as f:
            json.dump(self.index[index_name], f)

        print(f"Stored {index_name} index at {file_path}")

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        with open(path, 'r') as f:
            return json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        print('comparing indexes')
        
        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the descriptions
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()

        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

def main():
    with open(os.path.join(BASE_DIR, 'crawled.json'), 'r') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents")
    index = Index(docs)
    path = os.path.join(BASE_DIR, 'indexes') + os.sep
    for index_name in [
        Indexes.DOCUMENTS.value,
        Indexes.CHARACTERS.value,
        Indexes.GENRES.value,
        Indexes.DESCRIPTIONS.value,
    ]:
        index.store_index(path, index_name)
    index.check_if_indexing_is_good(Indexes.DESCRIPTIONS.value, 'good')
    index.check_if_indexing_is_good(Indexes.CHARACTERS.value, 'harry')
    index.check_if_indexing_is_good(Indexes.GENRES.value, 'fiction')

    index.check_add_remove_is_correct()
    loaded = index.load_index(f'{path}description_index.json')
    print(index.check_if_index_loaded_correctly(Indexes.DESCRIPTIONS.value, loaded))

if __name__ == '__main__':
    main()
