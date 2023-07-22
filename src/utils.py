import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.exception import CustomException
from src.logger import logging
import boto3

def save_object(file_path,obj):
    try:    
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception occured during load object function utils')
        raise CustomException(e,sys)
    
def S3_load_data(bucket_name_,object):

    try:
        s3 = boto3.client('s3')

        s3 = boto3.resource(
            service_name='s3',
            region_name='ap-south-1',
            aws_access_key_id='',
            aws_secret_access_key=''
        )

        bucket_name = bucket_name_
            
        obj = s3.Bucket(bucket_name).Object(object).get()

        df = pd.read_csv(obj['Body'])

        return df

    except Exception as e:
        logging.info('Exception occured during load dataset function utils')
        raise CustomException(e,sys)
