##########
# Import #
##############################################################################

import argparse
import logging
import joblib

import requests

from config.config import config
from framework import prediction, deep_prediction

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

def predict(end_point: str, entity: int, is_deep: bool):
    """Predict fuction

    Parameters
    ----------
    data : Union[dict, pd.DataFrame]
        If dict, convert to dataframe.
    model_type : str
        Assigned for the differeces data prepare logic.
    is_deep: bool
        if True, using deep prediction protocol.

    Raises
    ------
    ValueError("No data provided")
    
    """
    fe_store_payload = {
        "feature_service": config["FEATURE_TYPE"],
        "entity_rows": [
            {
                config["ENTITY_KEY"]: entity
            }
        ]
    }
    # request data from faeture store
    request_data = requests.get(end_point, json=fe_store_payload).json()
    logger.info("Request data success")
    
    if not request_data:
        raise ValueError("No data provided")
    
    # Prediction
    if is_deep is True:
        pred = deep_prediction(
            path = "model.pkl", 
            data = request_data,
            logger = logger,
        )[-1]
    elif is_deep is False:
        pred = prediction(
            path = "model.pkl", 
            data = request_data,
            logger = logger,
        )
    else:
        raise ValueError("is_deep need to be either True or False")
    
    logger.info(f"PREDICTION:{pred}")

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
    
    parser.add_argument(
        "is_deep", 
        help="If true, using deep prediction protocol",
        type=str2bool, 
        nargs='?', 
        const=True, 
        default=False,
    )
    
    return parser.parse_args()

##############################################################################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


##############################################################################

if __name__ == '__main__':
    args = parse_arguments()
    predict(args.endpoint, args.entity, args.is_deep)

##############################################################################
