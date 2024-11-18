##########
# Import #
##############################################################################

import pickle
import logging

import pandas as pd

from space_time_modeling.utilities import load_instance
from space_time_modeling.modeling.__classification_wrapper import ClassifierWrapper

#############
# Framework #
##############################################################################

def deep_prediction(
        path: str, 
        data: pd.DataFrame, 
        logger: logging.Logger,   
) -> tuple[float, str]:
    
    model_wrapper = load_instance(path)
    
    # Try and if error raise SystemError
    try:
        
        # Check null
        if data.isna().any().any():
            raise ValueError("Feature contain Null")

        logger.info(f"Model: {model_wrapper.name}")
        logger.info(f"Version: {model_wrapper.version}")
        logger.info(f"Feature: {model_wrapper.feature}")
        logger.info(f"Scraped data at: {data.iloc[[-1]]['scraped_timestamp'].values}")
        
        # Predict
        # Load model
        model: ClassifierWrapper = load_instance(path)
        pred = model(data)
        
        return pred, model_wrapper.version
        
    except SystemError:
        logger.error("Error at prediction")
        raise SystemError("Error at prediction")

##############################################################################

def evaluation(
        path: str, 
        data: pd.DataFrame, 
        logger: logging.Logger,   
) -> float:
    
    with open(path, "rb") as f:
        model_wrapper = pickle.load(f)
    
    # Try and if error raise SystemError
    try:
        
        # Check null
        if data.isna().any().any():
            raise ValueError("Feature containe Null")

        logger.info(f"Model: {model_wrapper.name}")
        logger.info(f"Version: {model_wrapper.version}")
        logger.info(f"Feature: {model_wrapper.feature}")
        
        # Predict
        metric = model_wrapper.evaluate(
            x_test=data,
        )
        
        return metric
        
    except SystemError:
        logger.error("Error at prediction")
        raise SystemError("Error at prediction")

##############################################################################
