import os
import sys
import numpy as np
import pandas as pd
import mlflow
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants import training_pipeline
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_pickle_obj, load_numpy_array, load_pickle_obj, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metrics
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import dagshub
dagshub.init(repo_owner='LILCOOTS', repo_name='NetworkSecurityML', mlflow=True)
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    # @staticmethod
    # def read_file(file_path):
    #     try:
    #         if not os.path.exists(file_path):
    #             raise Exception(f"The file: {file_path} is not present")
    #         return pd.read_csv(file_path)
    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys)

    
    def track_mlflow(self, best_model, classification_report):
        try:
            with mlflow.start_run():
                f1_score = classification_report.f1_score
                precision = classification_report.precision
                recall = classification_report.recall

                mlflow.log_metric("F1_Score", f1_score)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                
                # Log model parameters instead of the model object itself
                # since DagsHub doesn't support mlflow.sklearn.log_model()
                mlflow.log_param("model_type", type(best_model).__name__)
                if hasattr(best_model, 'get_params'):
                    model_params = best_model.get_params()
                    for param_name, param_value in model_params.items():
                        # Only log simple parameter types that MLflow can handle
                        if isinstance(param_value, (int, float, str, bool)) or param_value is None:
                            mlflow.log_param(f"model_{param_name}", param_value)


        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(verbose=1),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(verbose=1),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
                "AdaBoostClassifier": AdaBoostClassifier()
            }

            params = {
                "LogisticRegression": {},
                "DecisionTreeClassifier": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2" ]
                },

                "RandomForestClassifier": {
                    "n_estimators": [2,4,8,16,32,64,128],
                    "criterion": ["gini", "entropy"],
                    "max_features": ["sqrt", "log2"]    
                },

                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                },

                "GradientBoostingClassifier": {
                    "learning_rate": [0.1, 0.05, 0.01, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "loss": ["log_loss", "exponential"],
                    "criterion": ["friedman_mse", "squared_error"],
                    "n_estimators": [2,4,8,16,32,64,128],
                },

                "AdaBoostClassifier": {
                    "learning_rate": [0.1, 0.05, 0.01, 0.001],
                    "n_estimators": [2,4,8,16,32,64,128],
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, params=params)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.model_expected_accuracy:
                raise Exception(f"No best model found with expected accuracy: {self.model_trainer_config.model_expected_accuracy}. Model found is {best_model_name} with accuracy {best_model_score}")
            
            logging.info(f"Best found model on both training and testing dataset is: {best_model_name} with accuracy: {best_model_score}")

            train_classification_metric: ClassificationMetricArtifact = get_classification_metrics(y_true=y_train, y_pred=best_model.predict(X_train))

            # track the MLflow here
            self.track_mlflow(best_model=best_model, classification_report=train_classification_metric)

            test_classification_metric: ClassificationMetricArtifact = get_classification_metrics(y_true=y_test, y_pred=best_model.predict(X_test))

            # track the MLflow here
            self.track_mlflow(best_model=best_model, classification_report=test_classification_metric)

            preprocessor = load_pickle_obj(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            NetworkModelObj = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_pickle_obj(file_path=self.model_trainer_config.trained_model_file_path, obj = NetworkModelObj)

            final_dir = training_pipeline.MODEL_PUSHER_DIR_NAME
            os.makedirs(final_dir, exist_ok=True)
            final_preprocessor_path = os.path.join(final_dir, "model.pkl")
            save_pickle_obj(file_path=final_preprocessor_path, obj=best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_classification_metric,
                test_metric_artifact=test_classification_metric,
            )
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    def initiate(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model Trainer started")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)
            logging.info("Loaded transformed train and test array")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            logging.info("Model training completed")
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)