from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, save_object
import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR
from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from dataclasses import dataclass

logging.info('DATA TRANSFORMATION SCRIPT IS LOADING')

@dataclass
class DATA_TRANSFORMATION_CONFIG:
    preprocessing_path = os.path.join('artifacts', 'preprocessing.pkl')

logging.info('DATA TRANSFORMATION CONFIG CLASS DEFINED')

class DATA_TRANSFORMATION:
    def __init__(self):
        logging.info('CREATING OBJECT OF DATA_TRANSFORMATION_CONFIG')
        self.data_transformation_config_object = DATA_TRANSFORMATION_CONFIG()

    def feature_selection(self, train, test):
        try:
            logging.info('FEATURE SELECTION STAGE STARTED')

            keep_cols = [
                'family_history_diabetes',
                'physical_activity_minutes_per_week',
                'age',
                'bmi',
                'triglycerides',
                'ldl_cholesterol',
                'systolic_bp',
                'diet_score',
                'waist_to_hip_ratio',
                'hdl_cholesterol',
                'diagnosed_diabetes'
            ]

            train = train[keep_cols]
            test = test[keep_cols]

            logging.info('FEATURE SELECTION STAGE COMPLETED')
            logging.info(f'Train shape after feature selection: {train.shape}, '
                f'Test shape after feature selection: {test.shape}')

            return train, test

        except Exception as e:
            logging.error('ERROR OCCURRED DURING FEATURE SELECTION')
            raise CustomException(e, sys)

    def preprocessing(self, df: DataFrame):
        try:
            logging.info('PREPROCESSING STAGE STARTED')

            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            logging.info(f'Numerical columns detected: {numerical_columns}')
            logging.info(f'Categorical columns detected: {categorical_columns}')

            numerical_pipeline = Pipeline([
                ('imputer_numerical', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_pipeline = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'))])

            preprocessing = ColumnTransformer([
                ('numerical_pipeline_in_transformer',numerical_pipeline, numerical_columns),
                ('categorical_pipeline_in_transformer',categorical_pipeline, categorical_columns)])

            logging.info('PREPROCESSING OBJECT CREATED SUCCESSFULLY')

            return preprocessing

        except Exception as e:
            logging.error('ERROR OCCURRED DURING PREPROCESSING OBJECT CREATION')
            raise CustomException(e, sys)

    def transformation_method(self, train: DataFrame, test: DataFrame):
        try:
            logging.info('DATA TRANSFORMATION PIPELINE STARTED')

            logging.info('LOADING TRAIN AND TEST DATASETS')
            train_df = pd.read_csv(train)
            test_df = pd.read_csv(test)

            logging.info(f'Raw train shape: {train_df.shape}, '
                f'Raw test shape: {test_df.shape}')

            train_selected_columns_dataset, test_selected_columns_dataset = self.feature_selection(train_df, test_df)

            target = 'diagnosed_diabetes'

            train_feature = train_selected_columns_dataset.drop(columns=[target])
            train_target = train_selected_columns_dataset[target]

            test_feature = test_selected_columns_dataset.drop(columns=[target])
            test_target = test_selected_columns_dataset[target]

            logging.info('TARGET VARIABLE SEPARATED')

            preprocessing_object = self.preprocessing(train_feature)

            logging.info('APPLYING PREPROCESSING ON TRAIN DATA')
            train_feature_preprocessed = preprocessing_object.fit_transform(train_feature)

            logging.info('APPLYING PREPROCESSING ON TEST DATA')
            test_feature_preprocessed = preprocessing_object.transform(test_feature)

            train_array_concatenated = np.c_[train_feature_preprocessed, train_target]
            test_array_concatenated = np.c_[test_feature_preprocessed, test_target]

            logging.info('FEATURES AND TARGET CONCATENATED')

            save_object(self.data_transformation_config_object.preprocessing_path,preprocessing_object)

            logging.info(
                f'Preprocessing object saved at: '
                f'{self.data_transformation_config_object.preprocessing_path}')

            logging.info('DATA TRANSFORMATION PIPELINE COMPLETED SUCCESSFULLY')

            return train_array_concatenated, test_array_concatenated

        except Exception as e:
            logging.error('ERROR OCCURRED IN DATA TRANSFORMATION PIPELINE')
            raise CustomException(e, sys)
