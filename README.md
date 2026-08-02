# Goodreads Information Retrieval System

An end-to-end Information Retrieval pipeline built over a Goodreads dataset of the top 3000 rated books. It covers preprocessing, near-duplicate detection, inverted/tiered indexing, spell correction, multi-model scoring, ranked search, snippet generation, and evaluation, wrapped in a Streamlit search UI.

Built for the Modern Information Retrieval course (Sharif University of Technology).

## Features

- **Text preprocessing** — link/punctuation removal, case-folding, stopword filtering, and stemming (`Logic/preprocess.py`).
- **Near-duplicate detection** — MinHash + LSH over shingled descriptions to flag near-identical entries (`Logic/LSH.py`).
- **Inverted indexing** — separate indexes for `description`, `genres`, and `characters`, plus tiered, document-length, and metadata indexes for faster lookups (`Logic/indexer/`).
- **Spell correction** — Jaccard/k-gram candidate generation re-ranked by corpus term frequency (`Logic/spell_correction.py`).
- **Scoring models** — Vector Space Model (`ltn.lnn`, `ltc.lnc`), Okapi BM25, and Unigram Language Model with naive/Bayesian/mixture smoothing (`Logic/Scorer.py`).
- **Search engine** — safe and unsafe (tiered) ranking, per-field score aggregation with configurable weights, and genre preference boosting (`Logic/Search.py`).
- **Snippet generation** — extracts the highest-scoring text window around query terms and highlights matches with `***` markers, drawn from the original (non-normalized) text (`Logic/snippet.py`).
- **Evaluation** — MAP, NDCG, MRR, Precision, and Recall against a labeled query set (`Logic/Evaluation.py`).
- **Streamlit UI** — interactive search with highlighted snippets and result cards (`UI/main.py`).

## Project structure

```
mir/
├── Logic/
│   ├── preprocess.py          # Preprocessor, preprocess_docs, csv_to_json
│   ├── LSH.py                 # MinHashLSH near-duplicate detection
│   ├── Scorer.py              # VSM / BM25 / Unigram scoring
│   ├── Search.py               # SearchEngine (safe/unsafe ranking, aggregation)
│   ├── snippet.py             # Snippet extraction & highlighting
│   ├── spell_correction.py    # SpellCorrection
│   ├── Evaluation.py          # MAP / NDCG / MRR / Precision / Recall
│   ├── utils.py                # search() helper wired to SearchEngine + SpellCorrection
│   └── indexer/
│       ├── index.py                    # Index (build/store/load inverted indexes)
│       ├── tiered_index.py             # Tiered_index
│       ├── document_lengths_index.py   # DocumentLengthsIndex
│       ├── metadata_index.py           # Metadata_index
│       ├── index_reader.py             # Index_reader
│       └── indexes_enum.py             # Indexes / Index_types enums
├── UI/
│   └── main.py                 # Streamlit search app
├── indexes/                    # Prebuilt index JSON/pickle files
├── top_3000_rated_books.csv    # Raw dataset
├── top_3000_rated_books.rar    # Archived raw dataset
├── crawled.json                # Dataset converted to JSON
├── preprocessed.json           # Preprocessed dataset
├── stopwords.txt               # Stopword list used by the preprocessor
├── usage_eval.py               # Example script for running Evaluation
└── requirements.txt
```

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

The dataset is already included as `crawled.json` (and the raw `top_3000_rated_books.csv` / `.rar`). If you need to regenerate it from the CSV, use `csv_to_json` in `Logic/preprocess.py`.

### 3. Run the search UI

```bash
streamlit run UI/main.py
```

Then open `http://localhost:8501/` in your browser.

### 4. Run evaluation

```bash
python usage_eval.py
```

## Core concepts

### Preprocessing
`Preprocessor` strips URLs/emails, lower-cases and removes punctuation, tokenizes, filters stopwords (`stopwords.txt`), and stems remaining tokens before documents are indexed.

### Near-duplicate detection
`MinHashLSH` shingles each description, builds MinHash signatures, and buckets them into LSH bands so near-duplicate documents fall into the same bucket and can be filtered out. `Logic/LSHFakeData.json` is a sanity-check dataset where every consecutive pair is a known duplicate.

### Indexing
`Index` builds `{term: {doc_id: tf}}` postings for the `documents`, `description`, `genres`, and `characters` fields. `Tiered_index` splits postings into tiers by importance so unsafe ranking can scan only the top tiers first; `DocumentLengthsIndex` and `Metadata_index` store per-document length and corpus-level stats used by the scorers.

### Spell correction
`SpellCorrection` finds candidate corrections via k-gram Jaccard similarity against the corpus vocabulary, then re-scores candidates using normalized term frequency so common words are preferred over rare near-matches.

### Scoring
`Scorer` implements:
- **Vector Space Model** — `ltn.lnn` and `ltc.lnc` weighting schemes with cosine normalization.
- **Okapi BM25** — TF saturation with document-length normalization.
- **Unigram Language Model** — naive (`tf/dl`), Bayesian (Dirichlet-style), and mixture (linear interpolation with collection probabilities) smoothing.

### Search
`SearchEngine.search()` (exposed via `Logic/utils.py:search`) runs a query against the `characters`, `genres`, and `description` fields with configurable weights, merges per-field scores, and supports both safe (full postings) and unsafe (tiered) ranking, plus optional genre preference boosting.

### Snippets
`Snippet` finds the text window with the highest concentration of query terms for each document, merges adjacent windows with `...`, wraps matched terms in `***term***`, and reports any query terms missing from the document — matching is done on normalized text, but the returned snippet is pulled from the original raw text.

### Evaluation
`Evaluation` computes MAP, NDCG, MRR, Precision, and Recall by comparing ranked search results against a labeled relevance set.

## Requirements

See `requirements.txt`. Key dependencies: `streamlit`, `nltk`, `pandas`, `numpy`.
