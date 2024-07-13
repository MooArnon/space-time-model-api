##########
# Import #
##############################################################################

import pickle
import os
import json

import numpy as np
import pandas as pd
import requests

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import ClassificationModel 
from space_time_modeling.utilities import load_instance

#########
# Train #
##############################################################################

def train_model() -> None:
    # statics 
    label_column = "signal"
    id_columns = "id"
    
    data = {
            "feature_service": "complete_feature_label",
            "entities": "select id, current_timestamp as event_timestamp from feature_store.ma where asset = 'BTCUSDT'"
        }
    
    # Request data
    data = requests.get(
        url = "http://157.245.158.204:6000/feature/offline_feature/fetch",
        data = json.dumps(data)
    ).json()
    
    # Preprocess data
    df = pd.DataFrame(data=data).drop(columns=["event_timestamp", id_columns])
    df.dropna(inplace=True)
    feature_column = list(df.columns)
    feature_column.remove(label_column)
    # return df.columns
    # Train model
    modeling: ClassificationModel = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("classifier"),
        test_size = 0.15,
        n_iter = 15,
    )
    
    modeling.modeling(
        df = df, 
        model_name_list=['random_forest', 'knn', 'logistic_regression'],
    )
    
########
# Test #
##############################################################################

def test_model() -> None:
    
    # Data
    data = {
        "feature_service": "complete_feature",
        "entity_rows": [
            {
                "id": 6797
            }
        ]
    }

    data = requests.get(
        url = "http://157.245.158.204:6000/feature/online_feature/fetch",
        data = json.dumps(data)
    ).json()
    
    
    # Load model
    model = load_instance(
        "classifier_20240518_082212/catboost/catboost.pkl"
    )
    
    data_df = pd.DataFrame(data=data).drop(columns=["id"])[list(model.feature)]
    
    pred = model(data_df)
    
    print(pred)


#######
# Use #
##############################################################################

if __name__ == "__main__":
    
    train_model()
    # test_model()
    
    
    ##########################################################################

##############################################################################