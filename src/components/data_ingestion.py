from src.components.data_transformation import DataTransformation
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# ========================
# CONFIGURATION
# ========================
@dataclass
class DataIngestionConfig:
    data_path = os.path.join("notebook", "data", "Pune_UHI_Final_Dataset.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

   


# ========================
# DATA INGESTION CLASS
# ========================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Component for Pune UHI Project")

        try:
            # ------------------------
            # Load dataset
            # ------------------------
    
            df = pd.read_csv(self.ingestion_config.data_path)
            logging.info("Dataset loaded successfully")

            # ------------------------
            # Drop unnecessary columns
            # ------------------------
            drop_cols = ["system:index", ".geo"]
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])

            logging.info(f"Dropped columns: {drop_cols}")

            # ------------------------
            # Create artifacts folder
            # ------------------------
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # ------------------------
            # Save raw data
            # ------------------------
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # ------------------------
            # Train-test split
            # ------------------------
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # ------------------------
            # Save train & test data
            # ------------------------
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train-test split completed and saved")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ========================
# RUN PIPELINE
# ========================
if __name__ == "__main__":

    obj = DataIngestion()

    train_data_path, test_data_path = obj.initiate_data_ingestion()

    print("Data Ingestion Completed Successfully")

    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data_path,
        test_data_path
    )

    print("Data Transformation Completed Successfully")
    print("Preprocessor saved at:", preprocessor_path)