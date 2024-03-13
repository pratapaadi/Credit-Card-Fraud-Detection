import sys,os
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import classification_report,accuracy_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
def evaluate(X_train,y_train,X_test,y_test,model):
      try:
       report = {}
       for i in range(len(model)):
            model = list(model.values())[i]
            
            model.fit(X_train,y_train)
            y_test_pred =model.predict(X_test)
            model_report =classification_report(y_test,y_test_pred)

       return model_report
      except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
