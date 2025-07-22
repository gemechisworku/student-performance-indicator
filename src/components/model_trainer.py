import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'LinearRegression': LinearRegression()
            }
            params = {
                'RandomForestRegressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [.1, .01, .005, .001],
                    'subsample': [0.8, 0.9, 1.0],
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                'XGBRegressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [.1, .01, .005, .001],
                },
                'CatBoostRegressor': {
                    'iterations': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [5, 7, 9, 11,]
                },
                'DecisionTreeRegressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                'LinearRegression': {}
            }

            logging.info("Evaluating models")

            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model object saved successfully")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys) from e