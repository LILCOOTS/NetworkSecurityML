import os
import sys
import pandas as pd
import numpy as np
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constants import training_pipeline
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

# for checking data drift
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_file(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            col_count = len(self.schema_config)
            logging.info(f"Required number of columns: {col_count}")
            logging.info(f"Dataframe has columns: {len(df.columns)}")
            if len(df.columns) == col_count:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_data_drift(self, base_df, curr_df, threshold = 0.05) -> bool:
        try:    
            status = True
            report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = curr_df[col]
                p_value = ks_2samp(d1, d2).pvalue

                if threshold < p_value:
                    drift_found = False
                else:
                    drift_found = True
                    status = False
                report[col] = {
                    "p_value": float(p_value),
                    "drift_status": drift_found
                }
            logging.info("Drift information generated")
            # create directory to save report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok = True)
            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation has started")
            train_file_path = self.data_ingestion_artifact.train_file_path 
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the train and test file
            train_df = DataValidation.read_file(train_file_path)
            test_df = DataValidation.read_file(test_file_path)

            # validate the number of columns
            train_status = self.validate_number_of_columns(train_df)
            test_status = self.validate_number_of_columns(test_df)
            if not train_status:
                error_msg = "Train data does not have all the required columns"
            if not test_status:
                error_msg = "Test data does not have all the required columns"
            logging.info("Number of columns are validated successfully")

            # validate data drift
            status = self.detect_data_drift(base_df=train_df, curr_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_validation_config.valid_train_file_path,
                valid_test_file_path = self.data_validation_config.valid_test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

        