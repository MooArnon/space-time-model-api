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
from space_time_pipeline.nosql.dynamo_db import DynamoDB

from framework import prediction, deep_prediction, evaluation

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

model_category_mapping = {
    "random_forest": "classifier_model",
    "catboost": "classifier_model",
    "knn": "classifier_model",
    "logistic_regression": "classifier_model",
    "xgboost": "classifier_model",
    "lightgbm": "classifier_model",
    "dnn": "deep_classifier_model",
    "lstm": "deep_classifier_model",
    "rnn": "deep_classifier_model",
    "gru": "deep_classifier_model",
    "cnn": "deep_classifier_model",
    "dnn-short": "deep_classifier_model",
    "lstm-short": "deep_classifier_model",
    "rnn-short": "deep_classifier_model",
    "gru-short": "deep_classifier_model",
    "cnn-short": "deep_classifier_model",
}

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
    if bucket_name == "":
        bucket_name = os.getenv("BUCKET_NAME_MODEL")
    if prefix == "":
        prefix = os.getenv("PREFIX_MODEL")
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

def prediction_to_dynamo_db(
        asset: str,
        data_dict: dict,
        latest_id: str,
        model_type: str = "",
        bucket_name: str = "",
        prefix: str = "",
        table: str = "predictions"
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

    prediction_data = {}
    dynamo = DynamoDB()
    
    # Prediction
    pred, model_id = predict(
        data_dict=data_dict, 
        model_type=model_type, 
        bucket_name=bucket_name, 
        prefix=prefix,
    )
    logger.info(f"{model_id} predicted {pred}")
    
    pred_payload = {"value": pred, "confident": 1.0}
    
    prediction_data['asset'] = asset
    prediction_data['model_id'] = model_id
    prediction_data['model_type'] = model_type
    prediction_data['prediction'] = pred_payload
    prediction_data['record_type'] = 'MODEL'
    prediction_data['latest_id'] = latest_id
    prediction_data['predicted_timestamp'] = str(current_timestamp)

    prediction_data = dynamo.to_decimal(prediction_data)
    
    response = dynamo.ingest_data(table, item=prediction_data)
    
    logger.info(f"Done with response: {response}")

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
    local_file_path = f"/tmp/{model_type}.pkl"
    object_key = f'{prefix}{model_type}/{model_type}.pkl'
    
    model_category = model_category_mapping[model_type]

    # Upload the file with the prefix
    s3_client.download_file(bucket_name, object_key, local_file_path, )
    
    try:
        df = pd.DataFrame(data_dict)
        
        # Ensure the DataFrame contains the 'price' column and convert its type
        if 'price' not in df.columns:
            raise SystemError("DataFrame does not contain the 'price' column.")
        
        df = df.astype({'price': 'float64'})

        # Prediction
        if model_category == "classifier_model":
            pred, model_id = prediction(
                path=f"/tmp/{model_type}.pkl",
                data=df,
                logger=logger,
            )
        elif model_category == "deep_classifier_model":
            pred, model_id = deep_prediction(
                path=f"/tmp/{model_type}.pkl",
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

    # Call the main function with the assets list
    try:
        prediction_to_dynamo_db(
            asset = asset,
            data_dict = price_data,
            latest_id = latest_id,
            model_type = model_type,
            bucket_name = s3_bucket_model,
            prefix = prefix_model,
        )
        
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
                "0": "2024-07-23 08:00:03",
                "1": "2024-07-23 07:00:03",
                "2": "2024-07-23 09:00:03",
                "3": "2024-07-23 10:00:04",
                "4": "2024-07-23 11:00:02",
                "5": "2024-07-23 12:00:04",
                "6": "2024-07-23 13:00:03",
                "7": "2024-07-23 14:00:03",
                "8": "2024-07-23 15:00:03",
                "9": "2024-07-23 16:00:04",
                "10": "2024-07-23 17:00:04",
                "11": "2024-07-23 18:00:04",
                "12": "2024-07-23 19:00:02",
                "13": "2024-07-23 20:00:02",
                "14": "2024-07-23 21:00:02",
                "15": "2024-07-23 22:00:03",
                "16": "2024-07-23 23:00:03",
                "17": "2024-07-24 00:00:05",
                "18": "2024-07-24 01:00:03",
                "19": "2024-07-24 02:00:04",
                "20": "2024-07-24 03:00:03",
                "21": "2024-07-24 04:00:03",
                "22": "2024-07-24 05:00:03",
                "23": "2024-07-24 06:00:04",
                "24": "2024-07-24 07:00:03",
                "25": "2024-07-24 08:00:04",
                "26": "2024-07-24 09:00:03",
                "27": "2024-07-24 10:00:03",
                "28": "2024-07-24 11:00:04",
                "29": "2024-07-24 12:00:03",
                "30": "2024-07-24 13:00:03",
                "31": "2024-07-24 14:00:03",
                "32": "2024-07-24 15:00:03",
                "33": "2024-07-24 16:00:03",
                "34": "2024-07-24 17:00:04",
                "35": "2024-07-24 18:00:03",
                "36": "2024-07-24 19:00:03",
                "37": "2024-07-24 20:00:03",
                "38": "2024-07-24 21:00:03",
                "39": "2024-07-24 22:00:03",
                "40": "2024-07-24 23:00:04",
                "41": "2024-07-25 00:00:03",
                "42": "2024-07-25 01:00:04",
                "43": "2024-07-25 02:00:03",
                "44": "2024-07-25 03:00:03",
                "45": "2024-07-25 04:00:04",
                "46": "2024-07-25 05:00:04",
                "47": "2024-07-25 06:00:03",
                "48": "2024-07-25 07:00:03",
                "49": "2024-07-25 08:00:04",
                "50": "2024-07-25 09:00:03",
                "51": "2024-07-25 10:00:04",
                "52": "2024-07-25 11:00:04",
                "53": "2024-07-25 12:00:04",
                "54": "2024-07-25 13:00:04",
                "55": "2024-07-25 14:00:04",
                "56": "2024-07-25 15:00:03",
                "57": "2024-07-25 16:00:04",
                "58": "2024-07-25 17:00:03",
                "59": "2024-07-25 18:00:04",
                "60": "2024-07-25 19:00:04",
                "61": "2024-07-25 20:00:03",
                "62": "2024-07-25 21:00:04",
                "63": "2024-07-25 22:00:03",
                "64": "2024-07-25 23:00:03",
                "65": "2024-07-26 00:00:04",
                "66": "2024-07-26 01:00:03",
                "67": "2024-07-26 02:00:04",
                "68": "2024-07-26 03:00:04",
                "69": "2024-07-26 04:00:04",
                "70": "2024-07-26 05:00:04",
                "71": "2024-07-26 06:00:03",
                "72": "2024-07-26 07:00:03",
                "73": "2024-07-26 08:00:04",
                "74": "2024-07-26 09:00:05",
                "75": "2024-07-26 10:00:04",
                "76": "2024-07-26 11:00:03",
                "77": "2024-07-26 12:00:03",
                "78": "2024-07-26 13:00:03",
                "79": "2024-07-26 14:00:03",
                "80": "2024-07-26 15:00:03",
                "81": "2024-07-26 16:00:03",
                "82": "2024-07-26 17:00:01",
                "83": "2024-07-26 18:00:04",
                "84": "2024-07-26 19:00:03",
                "85": "2024-07-26 20:00:04",
                "86": "2024-07-26 21:00:03",
                "87": "2024-07-26 22:00:04",
                "88": "2024-07-26 23:00:02",
                "89": "2024-07-27 00:00:03",
                "90": "2024-07-27 01:00:04",
                "91": "2024-07-27 02:00:03",
                "92": "2024-07-27 03:00:03",
                "93": "2024-07-27 04:00:04",
                "94": "2024-07-27 05:00:02",
                "95": "2024-07-27 06:00:03",
                "96": "2024-07-27 07:00:02",
                "97": "2024-07-27 08:00:04",
                "98": "2024-07-27 09:00:03",
                "99": "2024-07-27 10:00:03",
                "100": "2024-07-27 11:00:04",
                "101": "2024-07-27 12:00:03",
                "102": "2024-07-27 13:00:03",
                "103": "2024-07-27 14:00:03",
                "104": "2024-07-27 15:00:02",
                "105": "2024-07-27 16:00:04",
                "106": "2024-07-27 17:00:04",
                "107": "2024-07-27 18:00:02",
                "108": "2024-07-27 19:00:04",
                "109": "2024-07-27 20:00:01",
                "110": "2024-07-27 21:00:03",
                "111": "2024-07-27 22:00:03",
                "112": "2024-07-27 23:00:05",
                "113": "2024-07-28 00:00:04",
                "114": "2024-07-28 01:00:04",
                "115": "2024-07-28 02:00:03",
                "116": "2024-07-28 03:00:03",
                "117": "2024-07-28 04:00:03",
                "118": "2024-07-28 05:00:04",
                "119": "2024-07-28 06:00:04",
                "120": "2024-07-28 07:00:03",
                "121": "2024-07-28 08:00:04",
                "122": "2024-07-28 09:00:04",
                "123": "2024-07-28 10:00:03",
                "124": "2024-07-28 11:00:03",
                "125": "2024-07-28 12:00:04",
                "126": "2024-07-28 13:00:02",
                "127": "2024-07-28 14:00:04",
                "128": "2024-07-28 15:00:03",
                "129": "2024-07-28 16:00:04",
                "130": "2024-07-28 17:00:04",
                "131": "2024-07-28 18:00:03",
                "132": "2024-07-28 19:00:03",
                "133": "2024-07-28 20:00:02",
                "134": "2024-07-28 21:00:03",
                "135": "2024-07-28 22:00:04",
                "136": "2024-07-28 23:00:03",
                "137": "2024-07-29 00:00:04",
                "138": "2024-07-29 01:00:01",
                "139": "2024-07-29 02:00:04",
                "140": "2024-07-29 03:00:04",
                "141": "2024-07-29 04:00:04",
                "142": "2024-07-29 05:00:04",
                "143": "2024-07-29 06:00:01",
                "144": "2024-07-29 07:00:04",
                "145": "2024-07-29 08:00:03",
                "146": "2024-07-29 09:00:03",
                "147": "2024-07-29 10:00:03",
                "148": "2024-07-29 11:00:04",
                "149": "2024-07-29 12:00:03",
                "150": "2024-07-29 13:00:03",
                "151": "2024-07-29 14:00:04",
                "152": "2024-07-29 15:00:03",
                "153": "2024-07-29 16:00:02",
                "154": "2024-07-29 17:00:03",
                "155": "2024-07-29 18:00:03",
                "156": "2024-07-29 19:00:03",
                "157": "2024-07-29 20:00:04",
                "158": "2024-07-29 21:00:04",
                "159": "2024-07-29 22:00:04",
                "160": "2024-07-29 23:00:04",
                "161": "2024-07-30 00:00:04",
                "162": "2024-07-30 01:00:03",
                "163": "2024-07-30 02:00:03",
                "164": "2024-07-30 03:00:04",
                "165": "2024-07-30 04:00:03",
                "166": "2024-07-30 05:00:02",
                "167": "2024-07-30 06:00:03",
                "168": "2024-07-30 07:00:04",
                "169": "2024-07-30 08:00:04",
                "170": "2024-07-30 09:00:03",
                "171": "2024-07-30 10:00:04",
                "172": "2024-07-30 11:00:04",
                "173": "2024-07-30 12:00:04",
                "174": "2024-07-30 13:00:04",
                "175": "2024-07-30 14:00:03",
                "176": "2024-07-30 15:00:03",
                "177": "2024-07-30 16:00:03",
                "178": "2024-07-30 17:00:02",
                "179": "2024-07-30 18:00:03",
                "180": "2024-07-30 19:00:03",
                "181": "2024-07-30 20:00:03",
                "182": "2024-07-30 21:00:02",
                "183": "2024-07-30 22:00:03",
                "184": "2024-07-30 23:00:03",
                "185": "2024-07-31 00:00:04",
                "186": "2024-07-31 01:00:03",
                "187": "2024-07-31 02:00:04",
                "188": "2024-07-31 03:00:04",
                "189": "2024-07-31 04:00:03",
                "190": "2024-07-31 05:00:03",
                "191": "2024-07-31 06:00:04",
                "192": "2024-07-31 07:00:04",
                "193": "2024-07-31 08:00:03",
                "194": "2024-07-31 09:00:05",
                "195": "2024-07-31 10:00:04",
                "196": "2024-07-31 11:00:02",
                "197": "2024-07-31 12:00:03",
                "198": "2024-07-31 13:00:03",
                "199": "2024-07-31 14:00:04",
                "200": "2024-07-31 15:00:03",
                "201": "2024-07-31 16:00:03",
                "202": "2024-07-31 17:00:04",
                "203": "2024-07-31 18:00:03",
                "204": "2024-07-31 19:00:03",
                "205": "2024-07-31 20:00:03",
                "206": "2024-07-31 21:00:04",
                "207": "2024-07-31 22:00:03",
                "208": "2024-07-31 23:00:04",
                "209": "2024-08-01 00:00:05",
                "210": "2024-08-01 01:00:03",
                "211": "2024-08-01 02:00:04",
                "212": "2024-08-01 03:00:04",
                "213": "2024-08-01 04:00:04",
                "214": "2024-08-01 05:00:04",
                "215": "2024-08-01 06:00:04",
                "216": "2024-08-01 07:00:04",
                "217": "2024-08-01 08:00:04",
                "218": "2024-08-01 09:00:04",
                "219": "2024-08-01 10:00:04",
                "220": "2024-08-01 11:00:04",
                "221": "2024-08-01 12:00:04",
                "222": "2024-08-01 13:00:03",
                "223": "2024-08-01 14:00:04",
                "224": "2024-08-01 15:00:04",
                "225": "2024-08-01 16:00:04",
                "226": "2024-08-01 17:00:04",
                "227": "2024-08-01 18:00:05",
                "228": "2024-08-01 19:00:03",
                "229": "2024-08-01 20:00:03",
                "230": "2024-08-01 21:00:03",
                "231": "2024-08-01 22:00:03",
                "232": "2024-08-01 23:00:04",
                "233": "2024-08-02 00:00:04",
                "234": "2024-08-02 01:00:04",
                "235": "2024-08-02 02:00:04",
                "236": "2024-08-02 03:00:04",
                "237": "2024-08-02 04:00:04",
                "238": "2024-08-02 05:00:04",
                "239": "2024-08-02 06:00:04",
                "240": "2024-08-02 07:00:04",
                "241": "2024-08-02 08:00:03",
                "242": "2024-08-02 09:00:03",
                "243": "2024-08-02 10:00:04",
                "244": "2024-08-02 11:00:04",
                "245": "2024-08-02 12:00:04",
                "246": "2024-08-02 13:00:04",
                "247": "2024-08-02 14:00:04",
                "248": "2024-08-02 15:00:04",
                "249": "2024-08-02 16:00:04",
                "250": "2024-08-02 17:00:04",
                "251": "2024-08-02 18:00:03",
                "252": "2024-08-02 19:00:04",
                "253": "2024-08-02 20:00:04",
                "254": "2024-08-02 21:00:03",
                "255": "2024-08-02 22:00:04",
                "256": "2024-08-02 23:00:02",
                "257": "2024-08-03 00:00:04",
                "258": "2024-08-03 01:00:04",
                "259": "2024-08-03 02:00:03",
                "260": "2024-08-03 03:00:04",
                "261": "2024-08-03 04:00:03",
                "262": "2024-08-03 05:00:03",
                "263": "2024-08-03 06:00:03",
                "264": "2024-08-03 07:00:04",
                "265": "2024-08-03 08:00:04",
                "266": "2024-08-03 09:00:04",
                "267": "2024-08-03 10:00:02",
                "268": "2024-08-03 11:00:04",
                "269": "2024-08-03 12:00:03",
                "270": "2024-08-03 13:00:03",
                "271": "2024-08-03 14:00:02",
                "272": "2024-08-03 15:00:04",
                "273": "2024-08-03 16:00:04",
                "274": "2024-08-03 17:00:04",
                "275": "2024-08-03 18:00:04",
                "276": "2024-08-03 19:00:04",
                "277": "2024-08-03 20:00:04",
                "278": "2024-08-03 21:00:03",
                "279": "2024-08-03 22:00:03",
                "280": "2024-08-03 23:00:03"
            },
            "price": {
                "0": 66548.4,
                "1": 66490.0,
                "2": 66859.4,
                "3": 66899.9,
                "4": 66843.3,
                "5": 66643.1,
                "6": 66478.3,
                "7": 66074.0,
                "8": 67192.8,
                "9": 66566.1,
                "10": 65819.4,
                "11": 65921.0,
                "12": 65840.1,
                "13": 65524.0,
                "14": 65810.3,
                "15": 65853.9,
                "16": 65896.0,
                "17": 65916.4,
                "18": 65763.2,
                "19": 65646.7,
                "20": 65857.3,
                "21": 66027.4,
                "22": 65834.0,
                "23": 65793.4,
                "24": 65874.2,
                "25": 66049.0,
                "26": 66412.5,
                "27": 66377.5,
                "28": 66476.6,
                "29": 66345.8,
                "30": 66216.0,
                "31": 66310.1,
                "32": 66735.0,
                "33": 66155.7,
                "34": 66370.1,
                "35": 66558.9,
                "36": 65733.4,
                "37": 65622.0,
                "38": 66028.0,
                "39": 65618.6,
                "40": 65276.8,
                "41": 65360.4,
                "42": 65482.7,
                "43": 64468.0,
                "44": 64190.8,
                "45": 64070.4,
                "46": 64203.9,
                "47": 64244.9,
                "48": 64302.6,
                "49": 64115.1,
                "50": 64327.7,
                "51": 64290.0,
                "52": 64141.9,
                "53": 64082.7,
                "54": 64106.5,
                "55": 64036.6,
                "56": 64809.9,
                "57": 64764.5,
                "58": 64854.1,
                "59": 65026.9,
                "60": 64545.8,
                "61": 64668.0,
                "62": 65282.0,
                "63": 65814.9,
                "64": 65617.9,
                "65": 65772.9,
                "66": 66360.0,
                "67": 66410.0,
                "68": 66961.5,
                "69": 67018.4,
                "70": 67045.3,
                "71": 66900.0,
                "72": 67030.4,
                "73": 66873.7,
                "74": 67076.3,
                "75": 67256.7,
                "76": 67270.3,
                "77": 67289.9,
                "78": 67277.1,
                "79": 67870.2,
                "80": 67295.7,
                "81": 67527.9,
                "82": 67367.6,
                "83": 67327.9,
                "84": 67542.7,
                "85": 67965.4,
                "86": 67402.8,
                "87": 67915.8,
                "88": 67900.7,
                "89": 67882.7,
                "90": 67847.1,
                "91": 67634.5,
                "92": 67995.9,
                "93": 67766.1,
                "94": 67839.1,
                "95": 67861.0,
                "96": 68055.5,
                "97": 68022.5,
                "98": 67962.1,
                "99": 68194.4,
                "100": 68125.2,
                "101": 68184.6,
                "102": 68371.0,
                "103": 69199.9,
                "104": 68952.5,
                "105": 68822.0,
                "106": 68662.2,
                "107": 68514.4,
                "108": 68529.3,
                "109": 68203.7,
                "110": 67755.2,
                "111": 68828.7,
                "112": 68600.0,
                "113": 67867.8,
                "114": 67924.9,
                "115": 68050.0,
                "116": 68117.0,
                "117": 67910.9,
                "118": 67315.3,
                "119": 67396.4,
                "120": 67459.9,
                "121": 67517.7,
                "122": 67351.0,
                "123": 67407.9,
                "124": 67533.9,
                "125": 67822.7,
                "126": 67961.5,
                "127": 67950.0,
                "128": 67707.1,
                "129": 67624.2,
                "130": 67646.2,
                "131": 67956.9,
                "132": 68202.2,
                "133": 68089.9,
                "134": 67985.9,
                "135": 67999.9,
                "136": 67977.1,
                "137": 68206.3,
                "138": 68671.6,
                "139": 68509.4,
                "140": 69527.3,
                "141": 69323.1,
                "142": 69349.1,
                "143": 69660.7,
                "144": 69553.1,
                "145": 69499.4,
                "146": 69436.0,
                "147": 69503.4,
                "148": 69583.1,
                "149": 69549.8,
                "150": 69775.9,
                "151": 69247.1,
                "152": 68202.1,
                "153": 68067.3,
                "154": 66906.0,
                "155": 66980.8,
                "156": 67389.8,
                "157": 67269.0,
                "158": 67348.3,
                "159": 67459.1,
                "160": 67186.6,
                "161": 66750.0,
                "162": 66572.0,
                "163": 66167.9,
                "164": 66391.8,
                "165": 66585.7,
                "166": 66778.2,
                "167": 66455.9,
                "168": 66733.2,
                "169": 66914.0,
                "170": 66737.4,
                "171": 66580.0,
                "172": 66545.6,
                "173": 66544.0,
                "174": 66635.3,
                "175": 66357.2,
                "176": 65779.3,
                "177": 66198.0,
                "178": 66150.1,
                "179": 65839.9,
                "180": 65800.2,
                "181": 65893.7,
                "182": 66141.0,
                "183": 66263.4,
                "184": 66192.0,
                "185": 66159.3,
                "186": 66308.9,
                "187": 66268.0,
                "188": 66081.8,
                "189": 66028.9,
                "190": 65774.1,
                "191": 66140.2,
                "192": 66425.4,
                "193": 66196.5,
                "194": 66285.2,
                "195": 66230.0,
                "196": 65965.8,
                "197": 66062.6,
                "198": 66128.8,
                "199": 66484.6,
                "200": 66408.9,
                "201": 66354.5,
                "202": 66326.3,
                "203": 66660.0,
                "204": 66495.1,
                "205": 65239.2,
                "206": 64556.0,
                "207": 64950.1,
                "208": 64943.3,
                "209": 64601.8,
                "210": 64630.5,
                "211": 64171.0,
                "212": 64081.1,
                "213": 63915.0,
                "214": 63716.9,
                "215": 64279.5,
                "216": 64204.5,
                "217": 64300.1,
                "218": 64470.5,
                "219": 64442.0,
                "220": 64555.9,
                "221": 64645.3,
                "222": 64856.0,
                "223": 64736.6,
                "224": 63918.2,
                "225": 62891.4,
                "226": 62882.0,
                "227": 63205.2,
                "228": 62935.6,
                "229": 63435.2,
                "230": 64764.0,
                "231": 64879.7,
                "232": 65168.1,
                "233": 65328.9,
                "234": 64871.3,
                "235": 64635.2,
                "236": 64339.8,
                "237": 64600.0,
                "238": 64520.6,
                "239": 64161.0,
                "240": 64457.0,
                "241": 64146.0,
                "242": 64574.0,
                "243": 64483.0,
                "244": 64730.6,
                "245": 64640.0,
                "246": 64828.3,
                "247": 65210.0,
                "248": 63136.8,
                "249": 63393.5,
                "250": 63289.9,
                "251": 63075.4,
                "252": 62874.9,
                "253": 62614.0,
                "254": 62597.9,
                "255": 61962.8,
                "256": 61379.8,
                "257": 61483.7,
                "258": 61130.0,
                "259": 61314.0,
                "260": 61417.5,
                "261": 61862.2,
                "262": 61544.2,
                "263": 61673.9,
                "264": 61804.0,
                "265": 61636.0,
                "266": 61605.8,
                "267": 61713.7,
                "268": 61787.3,
                "269": 61884.7,
                "270": 61969.1,
                "271": 61982.8,
                "272": 61963.9,
                "273": 60857.7,
                "274": 60892.7,
                "275": 60561.5,
                "276": 60221.2,
                "277": 60216.7,
                "278": 60375.0,
                "279": 60628.4,
                "280": 60598.5
            }
        },
        "s3_bucket_model": "space-time-model",
        "prefix_model":"classifier/btc/",
        "asset": "BTCUSDT",
        "limit": 50,
        "database": "warehouse",
        # "s3_bucket": "space-time-raw",
        # "prefix": "raw/prediction/classifier/to_be_processed"
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
