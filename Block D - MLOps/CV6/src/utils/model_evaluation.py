"""Evaluation metrics for model performance."""

from tensorflow import Tensor
from keras import backend as K


def f1(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculates the F1 score using precision and recall.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: The F1 score.
    """

    def recall_m(y_true: Tensor, y_pred: Tensor) -> None:
        """Calculates recall (sensitivity).

        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels.

        Returns:
            tensor: Recall score.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        FN = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        recall = TP / (TP + FN + K.epsilon())
        return recall

    def precision_m(y_true: Tensor, y_pred: Tensor) -> None:
        """Calculates precision.

        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels.

        Returns:
            tensor: Precision score.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        FP = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        precision = TP / (TP + FP + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
