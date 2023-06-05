# Basic Import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                input_feature_train_df,
                input_feature_test_df,
                target_feature_train_df,
                target_feature_test_df
            )
            # training with best model as analysed in EDA with hyper-tuning parameters

            best_model=RandomForestClassifier(n_estimators=200,max_depth=10,criterion='entropy',oob_score=True)
            best_model.fit(X_train,y_train)


            rf_predictions = best_model.predict(X_test)
            print("=================================================================")
            print("Random Forest Classifier:")
            print(classification_report(y_test, rf_predictions))
            print("=================================================================")

            logging.info("Random Forest Classifier Reprot : \n")
            logging.info(f"{classification_report(y_test, rf_predictions)}")
            
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)