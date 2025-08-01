from ast import Call
from typing import Callable
from metrics.BERTScore import compute_BertScore
from metrics.Bleu import compute_bleu
from metrics.RougeScore import compute_rouge
from metrics.Meteor import compute_meteor
from transformers import AutoTokenizer
from processing import *
def compute_metrics_factory(model_id:str, metrics_calls:list[Callable]=[compute_BertScore, compute_bleu, compute_rouge, compute_meteor]) -> Callable:
    """
    *'Light'* factory class implemented like function. The factory produce 'Callable'
    that perform exaustive metrics computation order to evaluate generative text2text
    models like LLMs.

    Metrics computed are:
        - BLEU
        - BERTScore
        - ROUGE
        - METEOR 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def _compute_metrics(eval_pred):
        # Preprocessing and decode tokens for all metrics
        _, decoded_preds_raw, decoded_labels_raw = decode_predictions_and_labels(eval_pred, tokenizer)
        processed_preds, reference = postprocess_text(decoded_preds_raw, decoded_labels_raw)
        flat_references = [ref[0] for ref in reference]

        results = {}
        
        for metric in metrics_calls:
            results.update( metric(processed_preds, flat_references)) # 
        

        # round results
        results = {k: float(round(v, 4)) for k, v in results.items() if isinstance(v, (int, float))}
        return results
    

    
    return _compute_metrics


        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    model_id = "openai-community/gpt2"

    compute_metrics = compute_metrics_factory(model_id)
    tk = AutoTokenizer.from_pretrained(model_id)

    target = [
        "Baronne James de Rothschild-nee Betty von Rothschild | Jean-Auguste-Dominique Ingres | oil painting",
        "The quick brown fox jumps over the lazy dog.",
        "A man is walking in the park with his dog."
    ]
    pred = [
        "Baronne James de Rothschild-nee Betty von Rothschild | photo", # Simile al target 1
        "The brown fox jumps over the quick lazy cat.",                # Alcune differenze
        "A person walks in the park with a pet."                      # Maggiore differenza
    ]

    if tk.pad_token is None:
        tk.pad_token = tk.eos_token # Usa il token di fine sequenza come pad_token
    pred_tokens = tk.batch_encode_plus(pred, padding='longest', return_tensors='np')["input_ids"]
    target_tokens = tk.batch_encode_plus(target, padding='longest', return_tensors='np')["input_ids"]

    print(compute_metrics((pred_tokens,target_tokens)))