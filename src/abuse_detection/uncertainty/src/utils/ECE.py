import torch
from torchmetrics.classification import BinaryCalibrationError

def compute_multilabel_ece(
    y_prob,
    y_true,
    n_bins=10,
    norm="l1",
    label_names=["방임", "정서", "신체", "성"]
):
    """
    y_prob: (샘플수, 라벨개수) sigmoid probabilities
    y_true: (샘플수, 라벨개수) binary labels
    """

    y_prob = torch.tensor(y_prob, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.int64)

    n_labels = y_prob.shape[1]
    ece_per_label = {}

    for i in range(n_labels):
        metric = BinaryCalibrationError(n_bins=n_bins, norm=norm)
        ece = float(metric(y_prob[:, i], y_true[:, i]))

        label = label_names[i]
        ece_per_label[label] = ece

    values = list(ece_per_label.values())
    macro_ece = sum(values) / len(values)

    return ece_per_label, macro_ece