import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Logic.utils import search
from Logic.Evaluation import Evaluation

TEST_CASES = [
    {
        "query": "harry potter wizard school magic",
        "relevant_ids": ["4255", "3", "4256", "5", "15878", "4", "6", "2", "2005", "1"],
    },
    {
        "query": "detective mystery murder investigation",
        "relevant_ids": ["6943", "3441", "3437", "3439", "1618", "16043", "16042", "16041", "6853", "13145"],
    },
    {
        "query": "vampire dracula dark romance",
        "relevant_ids": ["12024", "17236", "17239", "17241", "19135", "18128", "5414"],
    },
]

MAX_RESULTS = 10
METHOD = "ltn.lnn"


def main():
    actual = []
    predicted = []

    for case in TEST_CASES:
        results = search(case["query"], MAX_RESULTS, method=METHOD)
        predicted_ids = [doc_id for doc_id, _score in results]

        print(f"Query: {case['query']!r}")
        print(f"  relevant (ground truth): {case['relevant_ids']}")
        print(f"  retrieved (top {MAX_RESULTS}):  {predicted_ids}")
        print()

        actual.append(case["relevant_ids"])
        predicted.append(predicted_ids)

    evaluator = Evaluation("usage_eval_demo")
    evaluator.calculate_evaluation(actual, predicted)


if __name__ == "__main__":
    main()
