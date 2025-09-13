import os
import sys
import json
import pandas as pd
import numpy as np
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
import pymongo
from dotenv import load_dotenv
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# certify is a package to handle SSL certificate verification issues with pymongo
import certifi
# .where() return the path to the certificate bundle file
ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(inplace=True, drop=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_to_mongo(self, database, collection, records):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGODB_URI)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

# if __name__ == "__main__":
#     FILE_PATH = "network_data/Phishing_Websites_Data.csv"
#     DATABASE = "NetworkDB"
#     COLLECTION = "NetworkData"
#     networkDataExtractObj = NetworkDataExtract()
#     records = networkDataExtractObj.csv_to_json_converter(FILE_PATH)
#     print(networkDataExtractObj.insert_data_to_mongo(DATABASE, COLLECTION, records))