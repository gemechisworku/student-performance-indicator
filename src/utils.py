import os
import sys
import src.exception as CustomException
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