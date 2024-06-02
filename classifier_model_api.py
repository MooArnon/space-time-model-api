##########
# Import #
##############################################################################

import argparse
import logging
import joblib

import requests

from config.config import config
from framework import prediction

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

def predict(end_point: str, entity: int):
    """Predict fuction

    Parameters
    ----------
    data : Union[dict, pd.DataFrame]
        If dict, convert to dataframe
    model_type : str
        Assigned for the differeces data prepare logic

    Raises
    ------
    SystemError
        If any error occure
    """
    fe_store_payload = {
        "feature_service": config["FEATURE_TYPE"],
        "entity_rows": [
            {
                config["ENTITY_KEY"]: entity
            }
        ]
    }
    try:
        # request data from faeture store
        request_data = requests.get(end_point, json=fe_store_payload).json()
        logger.info("Request data success")
        
        if not request_data:
            raise ValueError("No data provided")
        
        # Prediction
        pred = prediction(
            path = "model.pkl", 
            data = request_data,
            logger = logger,
        )
        logger.info(f"PREDICTION:{pred}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make predictions using an API endpoint"
    )
    parser.add_argument(
        "endpoint", 
        type=str, 
        help="URL of the API endpoint",
    )
    parser.add_argument(
        "entity", 
        type=int, 
        help="Entity value to be used in the prediction",
    )
    return parser.parse_args()

##############################################################################

if __name__ == '__main__':
    args = parse_arguments()
    predict(args.endpoint, args.entity)

##############################################################################
