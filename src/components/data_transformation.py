import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation has started")

            #Define which columns to ordinal encode and which items to scale
            categorical_cols = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines']
            numerical_cols = ['votes', 'cost for two']

            logging.info("Pipeline Initiated")

            #Numerical Pipeline
            '''
            1) Handle Missing values
            2) Scaling
            '''
            numerical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')), #Median since there are outliers
                ]
            )

            #Categorical Piplepine
            '''
            1) Handle Missing values
            2) Label Encoding
            3) Scaling
            '''
            categorical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder()),
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_cols),
                ('categorical_pipeline',categorical_pipeline,categorical_cols)
            ],remainder='passthrough')

            return preprocessor
        
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error occured in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train and Test dataset complete")

            logging.info(f"Read Train Dataset \n {train_df.head().to_string()}")

            logging.info(f"Read Test Dataset \n {test_df.head().to_string()}")


            # #Rate 
            # train_df = train_df['rate'].apply(lambda x: float((str(x).split('/')[0] if (len(str(x))>3) else np.nan)))
            # test_df = test_df['rate'].apply(lambda x: float((str(x).split('/')[0] if (len(str(x))>3) else np.nan)))

            # #cost
            
            # train_df = train_df['cost for two'].str.replace(',','').astype(float)
            # #train_df['cost for two'] = train_df['cost for two'].apply(lambda x: str(x).replace(',','') if type(x) != float else float(x))
            # test_df = test_df['cost for two'].str.replace(',','').astype(float)
            
            

            # logging.info('obtaining preprocessing object')

            # preprocessing_obj = self.get_data_transformation_object()

            # target_column_name = 'rate'
            # drop_columns = [target_column_name]

            # input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            # target_feature_train_df = train_df[target_column_name]

            # input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            # target_feature_test_df = test_df[target_column_name]

            # #Transformation using preprocessor object
            # logging.info("Applying preprocessing object on train and test")

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            # test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            # save_object(
            #     file_path = self.data_transformation_config.preprocessor_obj_file_path,
            #     obj = preprocessing_obj
            # )
            # logging.info('Preprocessor Dill file Saved')
            
            # return(
            #     train_arr,
            #     test_arr,
            #     self.data_transformation_config.preprocessor_obj_file_path
                
            # )
            logging.info('Concatenating Train and test csvs')

            df = pd.concat([train_df,test_df],axis=0)

            
            
            logging.info('Obtaining preprocessing object')

            
            #Rate handling
            df['rate'] = df['rate'].apply(lambda x: float(x.split('/')[0]) if (len(x)>3) else 0)
            

            #Cost handling
            df['cost for two'] =df['cost for two'].str.replace(',','').astype(float)

            logging.info(f'DataFrame Head: \n {df.head().to_string()}')

            features = df.drop(['rate'],axis=1)
            target = df['rate']

            #Applying these pipelines on dataset specifically
            preprocessor_obj = self.get_data_transformation_object()

            features = preprocessor_obj.fit_transform(features)
            # test_df1 = preprocessor_obj.fit_transform(test_df)
 
            num_columns = ['votes', 'cost for two']
            cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines']            
            all_columns = num_columns+cat_columns

            features = pd.DataFrame(features, columns = all_columns)

            
            logging.info(f'DataFrame Head: \n {df.head().to_string()}')
            
            
            logging.info("Applying preprocessing object on training and testing datasets.")
 
            #Saving the preprocessor.pkl object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            # logging.info("preprocessor dill file saved.")
            logging.info('All sort of transformation has been done.')
            return (
                features,
                target,
                self.data_transformation_config.preprocessor_obj_file_path   
            )
        except Exception as e:
            logging.info('Error occured in initiate Data Transformation')
            raise CustomException(e,sys)