##########
# Import #
##############################################################################

import logging
import joblib

from flask import Flask, request

from framework import prediction

###########
# Statics #
##############################################################################

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
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
    
    request_data = request.get_json()
    
    if not request_data:
        raise ValueError("No data provided")
    
    data = request_data.get('data')
    model_type = request_data.get('model_type')
        
    pred = prediction(
        path = "model.pkl", 
        data = data,
        logger = logger,
        model_type = model_type,
    )
    
    return {
        "prediction": pred
    }

##############################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

##############################################################################
