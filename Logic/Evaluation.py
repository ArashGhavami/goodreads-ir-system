import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        total_precision = 0.0
        n = len(actual)
        for act, pred in zip(actual, predicted):
            if not pred:
                continue
            relevant_retrieved = set(act).intersection(set(pred))
            precision = len(relevant_retrieved) / len(pred)
            total_precision += precision
        return total_precision / n if n > 0 else 0.0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        total_recall = 0.0
        n = len(actual)
        
        for act, pred in zip(actual, predicted):
            if not act:
                continue
            
            relevant_retrieved = set(act).intersection(set(pred))
            recall = len(relevant_retrieved) / len(act)
            total_recall += recall
            
        return total_recall / n if n > 0 else 0.0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)
        
        if (p + r) == 0:
            return 0.0
        
        return 2 * (p * r) / (p + r)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        if not actual:
            return 0.0
            
        score = 0.0
        num_hits = 0
        actual_set = set(actual)
        
        for i, p in enumerate(predicted):
            if p in actual_set:
                num_hits += 1
                score += num_hits / (i + 1)
        return score / len(actual)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        scores = [self._average_precision_single(a, p) for a, p in zip(actual, predicted)]
        return sum(scores) / len(scores) if scores else 0.0

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        score = 0.0
        actual_set = set(actual)
        for i, p in enumerate(predicted):
            if p in actual_set:
                score += 1.0 / math.log2(i + 2)
        return score

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        scores = [self._dcg_single(a, p) for a, p in zip(actual, predicted)]
        return sum(scores) / len(scores) if scores else 0.0

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        total_ndcg = 0.0
        n = len(actual)
        
        for act, pred in zip(actual, predicted):
            dcg = self._dcg_single(act, pred)
            idcg = self._dcg_single(act, act)
            
            if idcg > 0:
                total_ndcg += (dcg / idcg)
                
        return total_ndcg / n if n > 0 else 0.0

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        total_rr = 0.0
        for act, pred in zip(actual, predicted):
            act_set = set(act)
            for i, p in enumerate(pred):
                if p in act_set:
                    total_rr += 1.0 / (i + 1)
                    break
        return total_rr

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        rr_sum = self.cacluate_RR(actual, predicted)
        n = len(actual)
        return rr_sum / n if n > 0 else 0.0
        
    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        m_ap = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        print(f"Evaluation Results:")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"MAP:       {m_ap:.4f}")
        print(f"NDCG:      {ndcg:.4f}")
        print(f"MRR:       {mrr:.4f}")

        self.log_evaluation(p, r, f1, ap, m_ap, dcg, ndcg, rr, mrr)
        
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "ap": ap,
            "map": m_ap,
            "dcg": dcg,
            "ndcg": ndcg,
            "rr": rr,
            "mrr": mrr
        }
