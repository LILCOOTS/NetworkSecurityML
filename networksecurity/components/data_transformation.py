import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.constants import training_pipeline
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_pickle_obj

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_file(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_data_transformer_object(self, cls) -> Pipeline:
        '''
        it initializes the KNNImputer and returns the pipeline object 

        cls: KNNImputer class
        return: Pipeline object

        '''
        try:
            logging.info("Initializing the KNNImputer")
            imputer: KNNImputer = KNNImputer(**training_pipeline.DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("Creating the KNNImputer pipeline object")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    
    def initiate(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            train_df = DataTransformation.read_file(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_file(self.data_validation_artifact.valid_test_file_path)
            logging.info("Read train and test data completed")

            # Seperating input feature and target feature
            train_input_feature_df = train_df.drop(columns=[training_pipeline.TARGET_COLUMN], axis=1)
            train_target_feature_df = train_df[training_pipeline.TARGET_COLUMN]
            train_target_feature_df.replace(to_replace=-1, value=0, inplace=True)
            
            test_input_feature_df = test_df.drop(columns=[training_pipeline.TARGET_COLUMN], axis=1)
            test_target_feature_df = test_df[training_pipeline.TARGET_COLUMN]
            test_target_feature_df.replace(to_replace=-1, value=0, inplace=True)
            logging.info("Separated input and target feature from both train and test data")

            # Transforming the input features
            preprocessor_obj = self.get_data_transformer_object(KNNImputer)
            preprocessor_obj.fit(train_input_feature_df)
            transformed_train_input_feature = preprocessor_obj.transform(train_input_feature_df)
            transformed_test_input_feature = preprocessor_obj.transform(test_input_feature_df)

            train_arr = np.c_[transformed_train_input_feature, np.array(train_target_feature_df)]
            test_arr = np.c_[transformed_test_input_feature, np.array(test_target_feature_df)]

            # Save the transformed data
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_pickle_obj(file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor_obj)

            # Prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
