import os
import re
import string
import json
import csv
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords


class Preprocessor:
    def __init__(self, custom_stopwords_path=None):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        if custom_stopwords_path is None:
            custom_stopwords_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'stopwords.txt'
            )
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_pattern = re.compile(pattern)
        self.stemmer = PorterStemmer()

        self.stopwords = set()
        try:
            with open(custom_stopwords_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.stopwords.add(word)
        except FileNotFoundError:
            pass 


    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.url_pattern.sub(' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.normalize(t) for t in tokens]
        return ' '.join(tokens)

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        if not text or not isinstance(text, str):
            return []

        tokens = text.lower().split()
        return [t for t in tokens if t not in self.stopwords]
        
    
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        list
            The normalized word.
        """
        return self.stemmer.stem(word.lower())
    

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        return [self.preprocess_text(doc) for doc in documents]

    

def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()
    for doc in docs:
        for field in ['title', 'description', 'author']:
            if field in doc:
                value = doc[field]
                if isinstance(value, list):
                    doc[field] = [preprocessor.preprocess_text(v) for v in value]
                elif isinstance(value, str):
                    doc[field] = preprocessor.preprocess_text(value)
                


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    books = []

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            book = {
                "id":           row.get("bookId", ""),
                "title":        row.get("title", ""),
                "author":       row.get("author", ""),
                "description":  row.get("description", ""),
                "genres":       [g.strip() for g in row.get("genres", "").split(',') if g.strip()],
                "characters":   [c.strip() for c in row.get("characters", "").split(',') if c.strip()],
                "languages":    [l.strip() for l in row.get("language", "").split(',') if l.strip()],
                "publish_date": row.get("publishDate", ""),
                "num_pages":    row.get("numPages", ""),
                "avg_rating":   row.get("rating", ""),
            }
            books.append(book)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(books, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_to_json(
        os.path.join(PROJECT_ROOT, 'top_3000_rated_books.csv'),
        os.path.join(PROJECT_ROOT, 'crawled.json'),
    )

    json_file_path = os.path.join(PROJECT_ROOT, 'crawled.json')
    with open(json_file_path, "r") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open(os.path.join(PROJECT_ROOT, 'preprocessed.json'), "w") as file:
        file.write(json.dumps(docs))
