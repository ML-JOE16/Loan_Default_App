import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

# Metrics to evaluate the model
from sklearn import metrics
# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                
            }
            params={
                
                "Decision Tree": {
                    #'criterion':['gini', 'entropy'],
                    'splitter':['best','random'],
                    "min_samples_leaf": [5, 10, 20, 50],
                    #'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    #'criterion':['gini', 'entropy'],
                    "n_estimators": [50,100, 150, 200],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_leaf": [5, 10, 20, 50],
                    "max_features": [0.8, 0.9],
                    "max_samples": [0.9, 1],
                    "class_weight": ["balanced",{0: 0.2, 1: 0.8}]
                    #'max_features':['sqrt','log2',None],
                },
                "Gradient Boosting":{
                    #'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':[None,'sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.4:
                raise CustomException("No best model found")
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1 = f1_score(y_test, predicted)
            acc = accuracy_score(y_test, predicted)
            recal = recall_score (y_test, predicted)
            return f1, acc, recal
            
            
        except Exception as e:
            raise CustomException(e,sys)