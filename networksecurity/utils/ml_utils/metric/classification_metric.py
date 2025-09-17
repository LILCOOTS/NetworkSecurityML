import sys, os
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        # model_accuracy = accuracy_score(y_true, y_pred)
        model_precision = precision_score(y_true, y_pred)
        model_recall = recall_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)

        classification_report = ClassificationMetricArtifact(precision=model_precision, recall=model_recall, f1_score=model_f1_score)
        return classification_report
    except Exception as e:
        raise NetworkSecurityException(e, sys)