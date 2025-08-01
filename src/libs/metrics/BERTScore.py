from ctypes import ArgumentError
import numpy as np
from typing import Any
from evaluate import load 

score  = load("bertscore")

def compute_BertScore(preds, ref) -> dict[Any, Any]|None:

    """Compute BertScore trained on english"""
    results = score.compute(predictions=preds, references=ref, lang="en")
    
    if not results:
        raise ArgumentError
    
    return {
        "bertscore_f1": np.mean(results["f1"]),
        "bertscore_precision": np.mean(results["precision"]),
        "bertscore_recall": np.mean(results["recall"]),
    }






