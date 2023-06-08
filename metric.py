import evaluate

roc_auc = evaluate.load('roc_auc', 'multilabel')

def compute_metric(eval_pred):
    logits, labels = eval_pred
    return roc_auc.compute(prediction_scores=logits, references=labels, average='samples')