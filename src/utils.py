import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import src.exception as CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    """
    This function saves an object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException.CustomException(e, sys) from e
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    This function evaluates multiple models and returns the best model based on R2 score.
    """
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(
                estimator=model,
                param_grid=param,
                cv=3
            )
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            print("Best params for model {}: {}".format(list(models.keys())[i], gs.best_params_))

            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException.CustomException(e, sys) from e
    
def load_object(file_path):
    """
    This function loads an object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException.CustomException(e, sys) from e