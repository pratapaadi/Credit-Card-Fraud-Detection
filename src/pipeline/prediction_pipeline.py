import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                 Time:float,
                 Amount:float,
                 TransactionMethod:float,
                 TransactionId:float,
                 Location:float,
                 TypeofCard:float,
                 Bank:float):
                 
        
        self.Time=Time
        self.Amount=Amount
        self.TransactionMethod=TransactionMethod
        self.TransactionId=TransactionId
        self.Location=Location
        self.TypeofCard=TypeofCard
        self.Bank = Bank
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Time':[self.Time],
                'Amount':[self.Amount],
                'TransactionMethod':[self.TransactionMethod],
                'TransactionId':[self.TransactionId],
                'Location':[self.Location],
                'TypeofCard':[self.TypeofCard],
                'Bank':[self.Bank],  
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
