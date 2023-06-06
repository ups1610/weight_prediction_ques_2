# Basic Import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path_revenue = os.path.join('artifacts','Revenue_model.pkl')
    trained_model_file_path_duration = os.path.join('artifacts','Informational_duration_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training_for_revenue(self,input_feature_train_df,input_feature_test_df,target_feature_train_df_revenue,target_feature_test_df_revenue):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, X_test, y_train, y_test = (
                input_feature_train_df,
                input_feature_test_df,
                target_feature_train_df_revenue,
                target_feature_test_df_revenue
            )
            # training with best model as analysed in EDA with hyper-tuning parameters

            best_model=RandomForestClassifier(n_estimators=50,max_depth=10,criterion='entropy')
            best_model.fit(X_train,y_train)


            rf_predictions = best_model.predict(X_test)

            print("===================== Model trained for revenue prediction ===================================")
            print("Random Forest Classifier:")
            # Calculate accuracy and confusion matrix
            accuracy = accuracy_score(y_test, rf_predictions)
            confusion = confusion_matrix(y_test, rf_predictions) 
            print(f"accuracy score : {accuracy}")
            print(f"confusion matrix : {confusion}") 
            logging.info(f"accuracy score : {accuracy}")
            logging.info(f"confusion matrix : {confusion}")
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path_revenue,
                 obj=best_model
            )    

            logging.info("Model trained successfully and saved")      
          
        except Exception as e:
            logging.info('Exception occured at Model Training of Revenue')
            raise CustomException(e,sys)


    def initate_model_training_for_duration(self,input_feature_train_df,input_feature_test_df,target_feature_train_df_duration,target_feature_test_df_duration):
            try:
                logging.info('Splitting Dependent and Independent variables from train and test data')
                X_train,X_test, y_train, y_test = (
                    input_feature_train_df,
                    input_feature_test_df,
                    target_feature_train_df_duration,
                    target_feature_test_df_duration
                )
                # training with best model as analysed in EDA with hyper-tuning parameters

                best_model=RandomForestRegressor(n_estimators=50,max_depth=10,criterion='friedman_mse',oob_score=True)
                best_model.fit(X_train,y_train)


                rf_predictions = best_model.predict(X_test)
                r2_square=evaluate_model(y_test,rf_predictions)
            
                print("======================== Model trained for Info. Duration ================================")
                print("R2 score",r2_square*100) 
                logging.info(f"Random Forest Regressor report : {r2_square*100}")
                print('='*35)
                print('\n')
                
                
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path_duration,
                    obj=best_model
                )  
                logging.info("model trained successfully and saved")

            except Exception as e:
                logging.info('Exception occured at Model Training of Informational Duration')
                raise CustomException(e,sys)    