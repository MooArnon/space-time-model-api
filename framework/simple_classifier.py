##########
# Import #
##############################################################################

import pandas as pd
import logging

from space_time_modeling.modeling import ClassifierWrapper
from space_time_modeling.utilities import load_instance

#############
# Framework #
##############################################################################

def prediction(
        path: str, 
        data: dict, 
        logger: logging.Logger, 
        model_type: str    
) -> float:
    # Load model warpper
    model_wrapper: ClassifierWrapper = load_instance(path)
    
    # Try and if error raise SystemError
    try:
        
        # initiate the data frame
        data_df = pd.DataFrame(data=data).drop(columns=["id"])\
            [list(model_wrapper.feature)]

        # Predict
        pred = model_wrapper(data_df)
        
        return pred
        
    except SystemError:
        logger.error(f"Error at {model_type}")
        raise SystemError(f"Error at {model_type}")

##############################################################################
