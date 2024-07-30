##########
# Import #
##############################################################################

import pickle
import os
import json

import numpy as np
import pandas as pd
import torch.nn as nn
import requests

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import DeepClassificationModel 
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
    modeling: DeepClassificationModel = modeling_engine(
        engine = "deep_classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("nn_100_it"),
        mode ='random_search',
        n_iter = 100,
        dnn_params_dict = {
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            'epochs':[10, 50, 100, 500, 1000],
            'criterion':[nn.BCELoss(), nn.HuberLoss()],
            'module__hidden_layers': [
                [4],
                [4, 4],
                [4, 4, 4],
                [8],
                [8, 8],
                [8, 16, 8],
                [16, 32, 16],
                [8, 16, 16, 8],
                [16, 32, 32, 16],
                [8, 16, 32, 16, 8],
            ],
            'module__dropout': [0.1, 0.15, 0.2, 0.25]
        }
    )
    
    modeling.modeling(
        df = df,
        model_name_list = ['dnn'],
        batch_size = 8,
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