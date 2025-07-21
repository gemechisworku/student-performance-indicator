import os
import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a data transformation pipeline.
        It handles both numerical and categorical features.
        """
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Data transformation pipeline created successfully")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process.
        It reads the training and testing data, applies the transformation,
        and saves the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded successfully for transformation")

            logging.info("Loading preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'math_score'
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data")

            # Fit and transform the training data
            input_features_train_transformed = preprocessor_obj.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_features_train_transformed, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_transformed, np.array(target_feature_test)]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessor_obj
            )
            logging.info("Saved preprocessor object")

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e   
