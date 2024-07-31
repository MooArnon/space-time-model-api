##########
# Import #
##############################################################################

import argparse
import logging
import os
import joblib

from space_time_pipeline.data_warehouse import PostgreSQLDataWarehouse

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
        pred = prediction(
            path="model.pkl",
            data=df,
            logger=logger,
        )

        logger.info(f"PREDICTION: {pred}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise SystemError(f"An error occurred during prediction: {e}")

##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make predictions using an API endpoint"
    )
    parser.add_argument(
        "asset", 
        type=str, 
        help="Symbol of asset",
    )
    return parser.parse_args()

##############################################################################

if __name__ == '__main__':
    args = parse_arguments()
    predict(args.asset)

##############################################################################
