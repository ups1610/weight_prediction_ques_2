import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self):
        try:
            
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(model_path)
            obj = CustomData("Male",23.0,1.80,77.0,"yes","no",2.0,3.0,"Sometimes","no",2.0,"no",2.0,1.0,"Frequently","Public_Transportation")
            data_scaled = obj.get_data_as_dataframe()
            pred=model.predict(data_scaled)
            print("============Prediction time=========================")
            Category = {
                1:'Normal_Weight',
                5:'Overweight_Level_I',
                6:'Overweight_Level_II',
                2:'Obesity_Type_I',
                0:'Insufficient_Weight',
                3:'Obesity_Type_II',
                4:'Obesity_Type_III'
            }
            print(f"The Person lie in : {Category[pred[0]]}")
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    ## pediction using sample data----
    def __init__(self,
                 Gender:str,
                 Age:float,
                 Height:float,
                 Weight:float,
                 family_history_with_overweight:str,
                 FAVC:str,
                 FCVC:float,
                 NCP:float,
                 CAEC:str,
                 SMOKE:object,
                 CH2O:float,
                 SCC:str,
                 FAF:float,
                 TUE:float,
                 CALC:str,
                 MTRANS:str):
        
        self.gender = Gender
        self.age = Age
        self.height = Height
        self.weight = Weight
        self.family_history_weight = family_history_with_overweight
        self.favc = FAVC
        self.fcvc = FCVC
        self.ncp = NCP
        self.caec = CAEC
        self.smoke = SMOKE
        self.ch20 = CH2O
        self.scc = SCC
        self.faf = FAF
        self.tue = TUE
        self.calc = CALC
        self.mtarns = MTRANS

    def get_data_as_dataframe(self):
        try:
             
            custom_data_input_dict = {
                 'Gender':[self.gender],
                 'Age':[self.age],
                 'Height':[self.height],
                 'Weight':[self.weight],
                 'family_history_with_overweight':[self.family_history_weight],
                 'FAVC':[self.favc],
                 'FCVC':[self.fcvc],
                 'NCP':[self.ncp],
                 'CAEC':[self.caec],
                 'SMOKE':[self.smoke],
                 'CH2O':[self.ch20],
                 'SCC':[self.scc],
                 'FAF':[self.faf],
                 'TUE':[self.tue],
                 'CALC':[self.calc],
                 'MTRANS':[self.mtarns]
            }
            df = pd.DataFrame(custom_data_input_dict)
            encoder = LabelEncoder()
            df['Gender'] = encoder.fit_transform(df['Gender'])
            df['family_history_with_overweight'] = encoder.fit_transform(df['family_history_with_overweight'])
            df['FAVC'] = encoder.fit_transform(df['FAVC'])
            df['CAEC'] = encoder.fit_transform(df['CAEC'])
            df['SMOKE'] = encoder.fit_transform(df['SMOKE'])
            df['SCC'] = encoder.fit_transform(df['SCC'])
            df['CALC'] = encoder.fit_transform(df['CALC'])
            df['MTRANS'] = encoder.fit_transform(df['MTRANS'])
            logging.info('Dataframe Gathered and encoded')
            logging.info(f"{df}")
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = PredictPipeline()
    obj.predict()        