import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
# from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path_1=os.path.join('artifacts','preprocessed_train_data.csv')
    preprocessor_obj_file_path_2=os.path.join('artifacts','preprocessed_test_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self,train_df,test_df):
        try:
            logging.info('Data Transformation initiated')
            # Preprocess the dataset
            encoder = LabelEncoder()
            # encode the training data
            train_df['Gender'] = encoder.fit_transform(train_df['Gender'])
            train_df['family_history_with_overweight'] = encoder.fit_transform(train_df['family_history_with_overweight'])
            train_df['FAVC'] = encoder.fit_transform(train_df['FAVC'])
            train_df['CAEC'] = encoder.fit_transform(train_df['CAEC'])
            train_df['SMOKE'] = encoder.fit_transform(train_df['SMOKE'])
            train_df['SCC'] = encoder.fit_transform(train_df['SCC'])
            train_df['CALC'] = encoder.fit_transform(train_df['CALC'])
            train_df['MTRANS'] = encoder.fit_transform(train_df['MTRANS'])
            train_df['NObeyesdad'] = encoder.fit_transform(train_df['NObeyesdad'])
            
            # encode the testing data
            test_df['Gender'] = encoder.fit_transform(test_df['Gender'])
            test_df['family_history_with_overweight'] = encoder.fit_transform(test_df['family_history_with_overweight'])
            test_df['FAVC'] = encoder.fit_transform(test_df['FAVC'])
            test_df['CAEC'] = encoder.fit_transform(test_df['CAEC'])
            test_df['SMOKE'] = encoder.fit_transform(test_df['SMOKE'])
            test_df['SCC'] = encoder.fit_transform(test_df['SCC'])
            test_df['CALC'] = encoder.fit_transform(test_df['CALC'])
            test_df['MTRANS'] = encoder.fit_transform(test_df['MTRANS'])
            test_df['NObeyesdad'] = encoder.fit_transform(test_df['NObeyesdad'])

            return train_df,test_df

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Removing duplicated data')
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)

            #Apply label encoder to the data
            logging.info("LabelEncode the data")

            df_train,df_test = self.get_data_transformation_object(train_df,test_df)
            

            target_column_name = 'NObeyesdad'
            drop_columns = [target_column_name]

            input_feature_train_df = df_train.drop(columns=drop_columns,axis=1)
            target_feature_train_df=df_train[target_column_name]

            input_feature_test_df=df_test.drop(columns=drop_columns,axis=1)
            target_feature_test_df=df_test[target_column_name]
            

            df_train.to_csv(self.data_transformation_config.preprocessor_obj_file_path_1,index=False,header=True)
            df_test.to_csv(self.data_transformation_config.preprocessor_obj_file_path_2,index=False,header=True)

            logging.info("Preprocessed data csv's file saved")

            return (
                input_feature_train_df,
                target_feature_train_df,
                input_feature_test_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path_1,
                self.data_transformation_config.preprocessor_obj_file_path_2
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        

