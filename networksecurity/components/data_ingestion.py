import os
import sys
import pymongo
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def export_df_from_mongodb(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            logging.info("Exporting data from MongoDB")
            self.mongo_client = pymongo.MongoClient(MONGODB_URI)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], axis=1, inplace=True)
            df.replace(to_replace="na", value=np.nan, inplace = True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_data_into_feature_store(self, df: pd.DataFrame):
        try:
            feature_store_file_path_name = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path_name)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path_name, index=False, header=True)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def split_data_into_train_test(self, df: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(df, test_size = self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split")
            logging.info("Exporting data into train and test file")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exporting of train and test file is completed")

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate(self):
        try:
            logging.info("Data Ingestion has started")
            df = self.export_df_from_mongodb()
            logging.info("Exporting data from MongoDB is completed")
            self.export_data_into_feature_store(df)
            logging.info("Exporting data into feature store is completed")
            self.split_data_into_train_test(df)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info("Data Ingestion is completed")
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)