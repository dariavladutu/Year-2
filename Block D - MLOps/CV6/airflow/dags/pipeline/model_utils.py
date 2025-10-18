"""Utility functions for model evaluation and saving.
This module contains functions for calculating F1 score and saving the model to a specified API path."""

import tensorflow as tf
from keras import backend as K
import shutil

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def save_model_to_api():
    model_path = "/models/latest_model.h5"
    dest = "/app/models/deployed_model.h5"
    shutil.copy(model_path, dest)
    print("✅ Model saved to API.")
