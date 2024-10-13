##########
# Import #
##############################################################################

import argparse
from datetime import datetime, timezone
import logging
import os
import json
import uuid
import traceback

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
        bucket_name: str = "",
        prefix: str = "",
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
    pred, model_id = predict(
        data_dict=data_dict, 
        model_type=model_type, 
        bucket_name=bucket_name, 
        prefix=prefix,
    )
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

def predict(data_dict: dict, model_type: str, bucket_name: str, prefix: str):
    """
    Predict function

    Parameters
    ----------
    data_dict: dict
        Dictionary of data
    model_type: str
        The type of model using for prediction
    bucket_name: str
        S3 bucket to store model
    prefix: str
        prefix to model

    Raises
    ------
    SystemError
        If any error occurs.
    """
    # Path to model
    local_file_path = f"{model_type}.pkl"
    object_key = f'{prefix}{model_type}/{model_type}.pkl'

    # Upload the file with the prefix
    s3_client.download_file(bucket_name, object_key, local_file_path, )
    
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
        traceback.print_exc()
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

def handler(event = None, context = None):
    """
    AWS Lambda handler function.
    
    event: a dict, must contain an 'assets' key with a list of asset symbols.
    context: Lambda context object (not used here).
    """
    logger.info("Handler function started.")
    file_path = None
    
    # Get the list of assets from the event input
    s3_bucket_model = event.get('s3_bucket_model', "")
    prefix_model = event.get('prefix_model', "")
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
            bucket_name = s3_bucket_model,
            prefix = prefix_model,
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
        traceback.print_exc()
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
    event = {
        "price_data": {
            "scraped_timestamp": {
                "0": "2024-09-07 15:00:14.000000",
                "1": "2024-09-07 16:00:16.000000",
                "2": "2024-09-07 17:00:16.000000",
                "3": "2024-09-07 18:00:16.000000",
                "4": "2024-09-07 19:00:16.000000",
                "5": "2024-09-07 20:00:16.000000",
                "6": "2024-09-07 21:00:16.000000",
                "7": "2024-09-07 22:00:16.000000",
                "8": "2024-09-07 23:00:16.000000",
                "9": "2024-09-08 00:00:16.000000",
                "10": "2024-09-08 01:00:16.000000",
                "11": "2024-09-08 02:00:16.000000",
                "12": "2024-09-08 03:00:16.000000",
                "13": "2024-09-08 04:00:16.000000",
                "14": "2024-09-08 05:00:16.000000",
                "15": "2024-09-08 06:00:16.000000",
                "16": "2024-09-08 07:00:16.000000",
                "17": "2024-09-08 08:00:16.000000",
                "18": "2024-09-08 09:00:16.000000",
                "19": "2024-09-08 10:00:16.000000",
                "20": "2024-09-08 11:00:15.000000",
                "21": "2024-09-08 12:00:15.000000",
                "22": "2024-09-08 13:00:15.000000",
                "23": "2024-09-08 14:00:15.000000",
                "24": "2024-09-08 15:00:15.000000",
                "25": "2024-09-08 16:00:15.000000",
                "26": "2024-09-08 17:00:15.000000",
                "27": "2024-09-08 18:00:15.000000",
                "28": "2024-09-08 19:00:15.000000",
                "29": "2024-09-08 20:00:15.000000",
                "30": "2024-09-08 21:00:15.000000",
                "31": "2024-09-08 22:00:15.000000",
                "32": "2024-09-08 23:00:15.000000",
                "33": "2024-09-09 00:00:15.000000",
                "34": "2024-09-09 01:00:15.000000",
                "35": "2024-09-09 02:00:15.000000",
                "36": "2024-09-09 03:00:15.000000",
                "37": "2024-09-09 04:00:15.000000",
                "38": "2024-09-09 05:00:15.000000",
                "39": "2024-09-09 06:00:15.000000",
                "40": "2024-09-09 07:00:15.000000",
                "41": "2024-09-09 08:00:15.000000",
                "42": "2024-09-09 09:00:15.000000",
                "43": "2024-09-09 10:00:15.000000",
                "44": "2024-09-09 11:00:15.000000",
                "45": "2024-09-09 12:00:15.000000",
                "46": "2024-09-09 13:00:15.000000",
                "47": "2024-09-09 14:00:15.000000",
                "48": "2024-09-09 15:00:16.000000",
                "49": "2024-09-09 16:00:15.000000"
            },
            "price": {
                "0": 54644.36,
                "1": 54788.01,
                "2": 54467.01,
                "3": 54118,
                "4": 54283.99,
                "5": 54414,
                "6": 54154,
                "7": 53915.44,
                "8": 53954,
                "9": 54169.99,
                "10": 54148.99,
                "11": 54275,
                "12": 54344,
                "13": 54400,
                "14": 54491.01,
                "15": 54341,
                "16": 54292.01,
                "17": 54440.77,
                "18": 54482,
                "19": 54447.99,
                "20": 54612.99,
                "21": 54548.01,
                "22": 54173,
                "23": 54394.01,
                "24": 54150,
                "25": 53675.99,
                "26": 53918.99,
                "27": 54511.12,
                "28": 54416.99,
                "29": 54360,
                "30": 54379.02,
                "31": 54512.74,
                "32": 54519.99,
                "33": 54862.01,
                "34": 55058,
                "35": 55093.54,
                "36": 55120.99,
                "37": 55126.77,
                "38": 54850.81,
                "39": 54654,
                "40": 54910,
                "41": 54970.73,
                "42": 55246,
                "43": 55152,
                "44": 55377.65,
                "45": 55276.01,
                "46": 55333.52,
                "47": 55702,
                "48": 54836.01,
                "49": 55389.28
            }
        },
        "s3_bucket_model": "space-time-model",
        "prefix_model":"classifier/btc/",
        "asset": "BTCUSDT",
        "limit": 50,
        "database": "warehouse",
        "s3_bucket": "space-time-raw",
        "prefix": "raw/prediction/classifier/to_be_processed"
    }
    handler(event=event)
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
