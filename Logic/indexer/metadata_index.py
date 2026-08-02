import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Logic.indexer.index_reader import Index_reader
from Logic.indexer.indexes_enum import Indexes, Index_types

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self, path):
        """
        Reads the documents.
        
        """
        return Index_reader(path, index_name=Indexes.DOCUMENTS).index

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            Indexes.CHARACTERS.value: self.get_average_document_field_length('characters'),
            Indexes.GENRES.value: self.get_average_document_field_length('genres'),
            Indexes.DESCRIPTIONS.value: self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        ans = 0
        if not self.documents:
            return 0

        for doc in self.documents.values():
            field = doc.get(where, [])
            if isinstance(field, list):
                ans += len(field)
            elif isinstance(field, str):
                ans += len(field.split())

        return ans / len(self.documents)

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)
 
if __name__ == "__main__":
    meta_index = Metadata_index(os.path.join(BASE_DIR, 'indexes') + os.sep)