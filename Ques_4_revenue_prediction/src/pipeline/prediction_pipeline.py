import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.preprocessing import LabelEncoder
import pandas as pd

## ======================== Prediction for Revenue ==============================
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self):
        try:
            
            model_path=os.path.join('artifacts','Revenue_model.pkl')
            model=load_object(model_path)
            obj = CustomData(0,0.0,0,10,627.500000,0.02,0.05,0.0,0.0,"Feb",3,3,1,4,"Returning_Visitor",True)
            data_scaled = obj.get_data_as_dataframe()
            pred=model.predict(data_scaled)
            print("============Prediction time=========================")
            Category = {
                0:'Revenue Not Generated',
                1:'Revenue Generated'
            }
            print(f"On the basis information : {Category[pred[0]]}")
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    ## pediction using sample data----
    def __init__(self,
                 Administrative:int,
                 Administrative_Duration:float,
                 Informational:int,
                 ProductRelated:int,
                 ProductRelated_Duration:float,
                 BounceRates:float,
                 ExitRates:float,
                 PageValues:float,
                 SpecialDay:float,
                 Month:object,
                 OperatingSystems:int,
                 Browser:int,
                 Region:int,
                 TrafficType:int,
                 VisitorType:object,
                 Weekend:bool):
        
        self.Administrative = Administrative
        self.Administrative_Duration = Administrative_Duration
        self.Informational = Informational
        self.ProductRelated = ProductRelated
        self.ProductRelated_Duration = ProductRelated_Duration
        self.BounceRates = BounceRates
        self.ExitRates = ExitRates
        self.PageValues = PageValues
        self.SpecialDay = SpecialDay
        self.Month = Month
        self.OperatingSystems = OperatingSystems
        self.Browser = Browser
        self.Region = Region
        self.TrafficType = TrafficType
        self.VisitorType = VisitorType
        self.Weekend = Weekend

    def get_data_as_dataframe(self):
        try:
             
            custom_data_input_dict = {
                 'Administrative':[self.Administrative],
                 'Administrative_Duration':[self.Administrative_Duration],
                 'Informational':[self.Informational],
                 'ProductRelated':[self.ProductRelated],
                 'ProductRelated_Duration':[self.ProductRelated_Duration],
                 'BounceRates':[self.BounceRates],
                 'ExitRates':[self.ExitRates],
                 'PageValues':[self.PageValues],
                 'SpecialDay':[self.SpecialDay],
                 'Month':[self.Month],
                 'OperatingSystems':[self.OperatingSystems],
                 'Browser':[self.Browser],
                 'Region':[self.Region],
                 'TrafficType':[self.TrafficType],
                 'VisitorType':[self.VisitorType],
                 'Weekend':[self.Weekend]
            }
            df = pd.DataFrame(custom_data_input_dict)
            encoder = LabelEncoder()
            df['Weekend'] = encoder.fit_transform(df['Weekend'])
            df['VisitorType'] = encoder.fit_transform(df['VisitorType'])
            df['Month'] = encoder.fit_transform(df['Month'])
            logging.info('Dataframe Gathered and encoded')
            logging.info(f"{df}")
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = PredictPipeline()
    obj.predict()        