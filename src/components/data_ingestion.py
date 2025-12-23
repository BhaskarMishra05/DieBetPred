import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame

logging.info(
    'STARTING DATA INGESTION STAGE. '
    'THIS WILL SPLIT THE RAW FILE INTO TRAINING AND TESTING FILES'
)

@dataclass
class DATA_INGESTION_CONFIG():
    raw_file_path = os.path.join('artifacts', 'raw.csv')
    train_file_path = os.path.join('artifacts', 'train.csv')
    test_file_path = os.path.join('artifacts', 'test.csv')

class DATA_INGESTION():
    def __init__(self):
        self.data_ingestion_config_object = DATA_INGESTION_CONFIG()

    def ingestion_method_for_data_spliting(self) -> DataFrame:
        try:
            logging.info('Entered data ingestion method')

            logging.info('Attempting to read raw CSV file')
            df = pd.read_csv(self.data_ingestion_config_object.raw_file_path)
            logging.info(f'Raw data loaded successfully with shape: {df.shape}')

            logging.info('Performing train-test split')
            training_dataset, testing_dataset = train_test_split(df,random_state=42,test_size=0.2)

            logging.info(f'Training dataset shape: {training_dataset.shape}')
            logging.info(f'Testing dataset shape: {testing_dataset.shape}')

            logging.info('Writing training dataset to disk')
            training_dataset.to_csv(self.data_ingestion_config_object.train_file_path,index=False)

            logging.info('Writing testing dataset to disk')
            testing_dataset.to_csv(self.data_ingestion_config_object.test_file_path,index=False)

            logging.info('Data ingestion completed successfully')

            return (
                self.data_ingestion_config_object.raw_file_path,
                self.data_ingestion_config_object.train_file_path,
                self.data_ingestion_config_object.test_file_path
            )

        except Exception as e:
            logging.error('Failure in data ingestion stage')
            raise CustomException(e, sys)
