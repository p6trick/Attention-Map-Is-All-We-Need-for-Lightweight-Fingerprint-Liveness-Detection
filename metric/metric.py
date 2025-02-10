from torchmetrics.functional.classification import binary_stat_scores

def metric(correct, loss, total, FP, FN, att_correct):
    