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
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('Exception has occured in prediction')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                votes:float,
                cost_for_two:float,
                online_order:bool,
                book_table:bool,
                location:str,
                rest_type:str,
                cuisines:str):
    

        self.votes = votes
        self.cost_for_two = cost_for_two
        self.online_order = online_order
        self.book_table = book_table
        self.location = location
        self.rest_type = rest_type
        self.cuisines = cuisines
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                'votes':[self.votes],
                'cost_for_two':[self.cost_for_two],
                'online_order':[self.online_order],
                'book_table':[self.book_table],
                'location':[self.location],
                'rest_type':[self.rest_type],
                'cuisines':[self.cuisines],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            # print(df)
            return df

        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)