import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
from src.utils import save_object
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig():
    model_trainer_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,features,target):
        try:
            logging.info('Defining Dependent and Independent features')
            X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            
            models = {
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'ElasticNet':ElasticNet(),
            'RandomForestRegressor': RandomForestRegressor(),
            'RandomForestRegressor_Tuned_1' : RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001),
            'ExtraTreeRegressor': ExtraTreesRegressor(),
            'ExtraTreeRegressor_Tuned' : ExtraTreesRegressor(n_estimators = 100),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'DecisionTreeRegressor_Tuned' : DecisionTreeRegressor(min_samples_leaf=.0001)
}

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.model_trainer_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)