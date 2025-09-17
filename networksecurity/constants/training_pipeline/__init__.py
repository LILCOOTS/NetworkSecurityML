import os
import sys
import numpy as np
import pandas as pd

'''Define all the constants here for the training pipeline'''
TARGET_COLUMN: str = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR_NAME: str = "Artifacts"
FILE_NAME: str = "Phishing_Websites_Data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_DIR_NAME: str = "data_schema"
SCHEMA_FILE_NAME: str = "schema.yaml"
SCHEMA_FILE_PATH: str = os.path.join(SCHEMA_DIR_NAME, SCHEMA_FILE_NAME)

PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

SAVED_MODEL_DIR_NAME: str = "saved_models"
SAVED_MODEL_FILE_NAME: str = "model.pkl"

MODEL_PUSHER_DIR_NAME: str = "final_models"

'''Data ingestion related constants'''
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "NetworkDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

'''Data validation related constants'''
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME: str = "validated"
DATA_VALIDATION_INVALID_DIR_NAME: str = "invalidated"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

'''Data transformation related constants'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME: str = "transformed_object"

#knn-imputer to replace missing values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

'''Model trainer related constants'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_MODEL_EXPECTED_ACCURACY: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05


