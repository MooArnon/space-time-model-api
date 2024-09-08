##########
# Import #
##############################################################################

import argparse
from datetime import datetime, timezone
import logging
import os
import joblib
import json
import uuid

from space_time_pipeline.data_warehouse import PostgreSQLDataWarehouse
from space_time_pipeline.data_lake import S3DataLake

from framework import prediction, evaluation

###########
# Statics #
##############################################################################

# Load the pre-trained machine learning model
path = joblib.load('model.pkl')

# Add logger object
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

#######
# API #
##############################################################################

def prediction_to_s3(
        asset: str,
        model_type: str,
        s3_bucket: str,
        prefix: str,
) -> None:
    current_timestamp = datetime.now(timezone.utc)
    current_timestamp_formatted = current_timestamp\
        .strftime(format="%Y%m%d_%H%M%S")
    
    # File name
    file_name = f"{model_type}_{asset}_{current_timestamp_formatted}.json"
    
    # Modify S3 path
    s3 = S3DataLake(logger = logger)
    
    # Prediction
    pred, model_id = predict(asset = asset)

    # Finalize payload
    prediction_payload = {
        "id":[str(uuid.uuid4())],
        "asset":[asset],
        "model_type":[model_type],
        "model_id":[model_id],
        "prediction":[pred],
        "predicted_timestamp":[str(current_timestamp)],
    } 
    
    with open(f"{file_name}", "w") as json_file:
        json.dump(prediction_payload, json_file, indent=4)
    
    s3.upload_to_data_lake(
        s3_bucket = s3_bucket,
        prefix = prefix,
        target_file = file_name,
    )
    
##############################################################################

def predict(asset: str):
    """
    Predict function

    Parameters
    ----------
    asset : str
        The asset for which the prediction is to be made.

    Raises
    ------
    SystemError
        If any error occurs.
    """
    try:
        select_raw_data = os.path.join("framework", "sql", "select_data.sql")
        sql = PostgreSQLDataWarehouse()

        # Fetching data
        df = sql.select(
            logger=logger,
            file_path=select_raw_data,
            replace_condition_dict={'<ASSET>': asset}
        )

        # Closing the SQL connection
        sql.close_connection()

        # Ensure the DataFrame contains the 'price' column and convert its type
        if 'price' not in df.columns:
            raise SystemError("DataFrame does not contain the 'price' column.")
        
        df = df.astype({'price': 'float64'})

        # Prediction
        pred, model_id = prediction(
            path="model.pkl",
            data=df,
            logger=logger,
        )

        logger.info(f"PREDICTION: {pred}")
        
        return pred, model_id
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise SystemError(f"An error occurred during prediction: {e}")

##############################################################################

def evaluate(asset: str, evaluation_period: int) -> None:
    try:
        select_raw_data = os.path.join(
            "framework", 
            "sql", 
            "select_evaluate.sql",
        )
        sql = PostgreSQLDataWarehouse()

        # Fetching data
        # 170 as one week of evaluation
        df = sql.select(
            logger=logger,
            file_path=select_raw_data,
            replace_condition_dict={
                '<ASSET>': asset,
                '<LIMIT>': evaluation_period,
            }
        )

        # Closing the SQL connection
        sql.close_connection()

        # Ensure the DataFrame contains the 'price' column and convert its type
        if 'price' not in df.columns:
            raise SystemError("DataFrame does not contain the 'price' column.")
        
        df = df.astype({'price': 'float64'})

        # Prediction
        metric = evaluation(
            path="model.pkl",
            data=df,
            logger=logger,
        )
        logger.info(f"METRIC: {metric}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise SystemError(f"An error occurred during prediction: {e}")

##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make predictions using an API endpoint"
    )
    
    parser.add_argument(
        "mode", 
        type=str, 
        help="Calling mode, predict or evaluate",
    )
    
    parser.add_argument(
        "asset", 
        type=str, 
        help="Symbol of asset",
    )
    
    parser.add_argument(
        "model_type", 
        type=str, 
        help="Type of model",
    )
    
    parser.add_argument(
        "--evaluation-period",  
        type=int,  
        default=170, 
        help="Integer of periods to use for evaluation (default: 170)"
    )
    
    parser.add_argument(
        "--s3-bucket",  
        type=str,  
        default="space-time-raw", 
        help="Bucket to upload raw data"
    )
    
    parser.add_argument(
        "--prefix",  
        type=str,  
        default="raw/prediction/classifier/to_be_processed", 
        help="Prefix to store raw file"
    )

    return parser.parse_args()

##############################################################################

if __name__ == '__main__':
    args = parse_arguments()
    
    if args.mode == 'evaluate':
        evaluate(args.asset, args.evaluation_period)
    elif args.mode == 'predict':
        prediction_to_s3(args.asset, args.model_type, args.s3_bucket, args.prefix)
    else:
        raise ValueError("Mode could be either predict or evaluate ")

##############################################################################
