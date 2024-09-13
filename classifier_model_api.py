##########
# Import #
##############################################################################

import argparse
from datetime import datetime, timezone
import logging
import os
import json
import uuid
import shutil

import boto3
import pandas as pd
from space_time_pipeline.data_warehouse import PostgreSQLDataWarehouse

from framework import prediction, evaluation

###########
# Statics #
##############################################################################

# Add logger object
# Create a logger object
logger = logging.getLogger(__name__)

# Configure the logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Set the logging level
logger.setLevel(logging.INFO)

# Modify S3 path
s3_client = boto3.client('s3')

#######
# API #
##############################################################################

def prediction_to_s3(
        asset: str,
        data_dict: dict,
        latest_id: str,
        model_type: str = "",
) -> None:
    """Predict result and push predicted payload to S3

    Parameters
    ----------
    asset : str
        Symbol of asset
    data_dict : dict
        Dictionary of data
    latest_id : str
        ID to map to scraped data
    model_type : str
        Type of model
    s3_bucket : str, optional
        Working bucket to push raw json, by default None
    prefix : str, optional
        Prefix for push json, by default None
    """
    if model_type == "":
        model_type = os.getenv("MODEL")
    current_timestamp = datetime.now(timezone.utc)
    current_timestamp_formatted = current_timestamp\
        .strftime(format="%Y%m%d_%H%M%S")
    
    # File name
    file_name = f"{model_type}_{asset}_{current_timestamp_formatted}.json"
    file_path = f"/tmp/{file_name}"
    
    # Prediction
    pred, model_id = predict(data_dict=data_dict, model_type=model_type)
    logger.info(f"{model_id} predicted {pred}")

    # Finalize payload
    prediction_payload = {
        "id":[str(uuid.uuid4())],
        "asset":[asset],
        "raw_data_id": latest_id,
        "model_type":[model_type],
        "model_id":[model_id],
        "prediction":[pred],
        "predicted_timestamp":[str(current_timestamp)],
    } 
    logger.info(f"Generated payload: {prediction_payload}")
    
    # Write payload to json and upload data to S3
    with open(file_path, "w") as json_file:
        json.dump(prediction_payload, json_file, indent=4)
    logger.info(f"Saved file at {file_path}")
    
    return file_path, file_name
    
##############################################################################

def predict(data_dict: dict, model_type: str):
    """
    Predict function

    Parameters
    ----------
    data_dict: dict
        Dictionary of data
    model_type: str
        The type of model using for prediction

    Raises
    ------
    SystemError
        If any error occurs.
    """
    try:
        df = pd.DataFrame(data_dict)
        
        # Ensure the DataFrame contains the 'price' column and convert its type
        if 'price' not in df.columns:
            raise SystemError("DataFrame does not contain the 'price' column.")
        
        df = df.astype({'price': 'float64'})

        # Prediction
        pred, model_id = prediction(
            path=f"{model_type}.pkl",
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

def upload_file_to_s3(file_path: str, bucket_name: str, s3_key: str) -> None:
    """
    Upload a file to an S3 bucket.
    
    Parameters
    ----------
    file_path : str
        The path to the file to be uploaded.
    bucket_name : str
        The name of the S3 bucket.
    s3_key : str
        The S3 key (path) where the file will be stored.
    """
    try:
        with open(file_path, 'rb') as file:
            s3_client.upload_fileobj(file, bucket_name, s3_key)
        logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        raise

#################
# Lambda handel #
##############################################################################

def handler(event, context):
    """
    AWS Lambda handler function.
    
    event: a dict, must contain an 'assets' key with a list of asset symbols.
    context: Lambda context object (not used here).
    """
    logger.info("Handler function started.")
    file_path = None
    
    # Get the list of assets from the event input
    asset = event.get('asset', "")
    price_data = event.get('price_data', {})
    latest_id = event.get('latest_id', {})
    model_type = event.get('model_type', "")
    s3_bucket = event.get('s3_bucket', "")
    prefix = event.get('prefix', "")
    
    # Call the main function with the assets list
    try:
        file_path, file_name = prediction_to_s3(
            asset = asset,
            data_dict = price_data,
            latest_id = latest_id,
            model_type = model_type,
        )
        os.listdir("/tmp")
        
        s3_key = f"{prefix}/{file_name}"
        upload_file_to_s3(file_path, s3_bucket, s3_key)
        
        logger.info("Uploaded file at data lake")
        return {
            "statusCode": 200,
            "body": "Prediction completed successfully."
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file {file_path} removed.")

#################
# Local running #
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
    pass
    """
    with open("price_data.json", "r") as json_file:
        data_dict = json.load(json_file)
    
    args = parse_arguments()
    
    if args.mode == 'evaluate':
        evaluate(args.asset, args.evaluation_period)
    elif args.mode == 'predict':
        prediction_to_s3(
            args.asset, 
            data_dict,
            args.model_type, 
            args.s3_bucket, 
            args.prefix,
        )
    else:
        raise ValueError("Mode could be either predict or evaluate ")
    """

##############################################################################
