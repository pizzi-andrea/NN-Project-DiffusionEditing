from typing import Any
import evaluate


sacrebleu_metric = evaluate.load("sacrebleu")
def compute_bleu(predictions, references) -> dict[Any, Any] | None:
    """Compute BLEU metric via SacreBLEU implementation."""

    references = [[item] for item in references]
    results =  sacrebleu_metric.compute(predictions=predictions, references=references)
    return {
        "BLEU" : results['score'] # type: ignore
    }
