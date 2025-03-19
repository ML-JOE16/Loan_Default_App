import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self,
               LOAN: int,
               MORTDUE: float,
               VALUE: float,
               REASON: str,
               JOB: str,
               YOJ: float,
               DEROG: float,
               DELINQ: float,
               CLAGE: float,
               NINQ: float,
               CLNO: float,
               DEBTINC: float):
        
        self.LOAN = LOAN

        self.MORTDUE = MORTDUE

        self.VALUE = VALUE

        self.REASON = REASON

        self.JOB = JOB

        self.YOJ = YOJ

        self.DEROG = DEROG

        self.DELINQ = DELINQ

        self.CLAGE = CLAGE

        self.NINQ = NINQ

        self.CLNO = CLNO

        self.DEBTINC = DEBTINC

    def get_data_as_data_frame(self):
        # will return input as a data frame to train model
        try:
            custom_data_input_dict = {
                "LOAN": [self.LOAN],
               "MORTDUE":[self.MORTDUE],
               "VALUE": [self.VALUE],
               "REASON": [self.REASON],
               "JOB": [self.JOB],
               "YOJ": [self.YOJ],
               "DEROG": [self.DEROG],
               "DELINQ": [self.DELINQ],
               "CLAGE": [self.CLAGE],
               "NINQ": [self.NINQ],
               "CLNO": [self.CLNO],
               "DEBTINC": [self.DEBTINC]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)