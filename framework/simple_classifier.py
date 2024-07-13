##########
# Import #
##############################################################################

import pickle
import logging

import pandas as pd

#############
# Framework #
##############################################################################

def prediction(
        path: str, 
        data: dict, 
        logger: logging.Logger,   
) -> float:
    
    with open(path, "rb") as f:
        model_wrapper = pickle.load(f)
    
    # Try and if error raise SystemError
    try:
        
        # initiate the data frame
        data_df = pd.DataFrame(data=data, index=[0]).drop(columns=["id"])\
            [list(model_wrapper.feature)]
        
        # Check null
        if data_df.isna().any().any():
            raise ValueError("Feature containe Null")

        # Predict
        pred = model_wrapper(data_df)
        
        return pred
        
    except SystemError:
        logger.error("Error at prediction")
        raise SystemError("Error at prediction")

##############################################################################
