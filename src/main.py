from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DATA_INGESTION
from src.components.data_transformation import DATA_TRANSFORMATION
import sys
import os

try:
    ingestion_objects = DATA_INGESTION()
    _, train, test = ingestion_objects.ingestion_method_for_data_spliting()
except Exception as e:
    raise CustomException(e, sys)

try:
    transformation_object = DATA_TRANSFORMATION()
    train_array, test_array = transformation_object.transformation_method(
        train= train, test= test
    )
except Exception as e:
    raise CustomException(e,sys)