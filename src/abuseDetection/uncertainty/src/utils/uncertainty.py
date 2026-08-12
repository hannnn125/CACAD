from scipy.stats import entropy
import numpy as np

def compute_entropy(probs):
    return float(entropy(probs))

def compute_MSP(probs):
    return float(probs.max())

def compute_mismatch(label, preds, entropies, msps):
    mismatch_entropies = [
        entropy
        for gt, pred, entropy in zip(label, preds, entropies)
        if gt != pred
    ]
    mismatch_msps = [
        msp
        for gt, pred, msp in zip(label, preds, msps)
        if gt != pred
    ]

    return {
        "entropy": max(mismatch_entropies) if mismatch_entropies else None,
        "msp": min(mismatch_msps) if mismatch_msps else None,
    }

