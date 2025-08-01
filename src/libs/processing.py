import numpy as np

def postprocess_text(preds, labels):
    """
   
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def decode_predictions_and_labels(eval_preds, tokenizer):
    """
    """
    preds_input, label_ids = eval_preds
    
    current_preds = preds_input
    if isinstance(current_preds, tuple):
        current_preds = current_preds[0]

    if hasattr(current_preds, "ndim") and current_preds.ndim == 3:
        current_preds_ids = np.argmax(current_preds, axis=-1)
    else:
        current_preds_ids = current_preds

    # Remove null ids
    processed_label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    # decode into chars
    decoded_labels_raw = tokenizer.batch_decode(processed_label_ids, skip_special_tokens=True)
    decoded_preds_raw = tokenizer.batch_decode(current_preds_ids, skip_special_tokens=True)

    return current_preds_ids, decoded_preds_raw, decoded_labels_raw
