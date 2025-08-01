from evaluate import load

# Carica l'oggetto della metrica ROUGE una sola volta.
rouge_metric = load("rouge")

def compute_rouge(predictions, flat_references):
    """Compute Rouge metrics *(R1,R2,RL and RLsum)*"""
    rouge_output = rouge_metric.compute(predictions=predictions, references=flat_references, use_stemmer=True)
    if rouge_output:
        return {
            "rouge1": rouge_output.get("rouge1", 0.0),
            "rouge2": rouge_output.get("rouge2", 0.0),
            "rougeL": rouge_output.get("rougeL", 0.0),
            "rougeLsum": rouge_output.get("rougeLsum", 0.0)
        }
    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

