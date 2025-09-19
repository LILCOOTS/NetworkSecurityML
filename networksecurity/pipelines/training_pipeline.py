import os
import sys
from networksecurity.logging.logger import logging
from networksecurity.constants import training_pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

class TrainingPipeline:
    def __init__(self):
        try:
            self.pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.pipeline_config)
            data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion_obj.initiate()
            logging.info("Data ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.pipeline_config)
            data_validation_obj = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation_obj.initiate()
            logging.info("Data validation completed")
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.pipeline_config)
            data_transformation_obj = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation_obj.initiate()
            logging.info("Data transformation completed")
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.pipeline_config)
            model_trainer_obj = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer_obj.initiate()
            logging.info("Model trainer completed")
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)