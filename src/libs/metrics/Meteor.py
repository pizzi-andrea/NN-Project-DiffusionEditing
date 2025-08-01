import evaluate

meteor_metric = evaluate.load("meteor")
def compute_meteor(predictions, flat_references):
    """Compute METEOR metric given pred tokens and targed tokens"""
    return meteor_metric.compute(predictions=predictions, references=flat_references)
   
