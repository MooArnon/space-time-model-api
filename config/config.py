##########
# Import #
##############################################################################

import os

############
# Statics #
##############################################################################

config = {}
config["MODEL_TYPE_MAPPING"] = {
    "random_forest": "classifier_model",
    "catboost": "catboost_model",
    "knn": "classifier_model",
    "logistic_regression": "classifier_model",
    "xgboost": "xgboost_model",
    "dnn": "deep_classifier_model",
    "rnn": "deep_classifier_model",
}

config["FEATURE_TYPE"] = "complete_feature"
config["ENTITY_KEY"] = "id"

#######
# Env #
##############################################################################

config['DB_HOST'] = os.getenv('DB_HOST')
config['DB_USERNAME'] = os.getenv('DB_USERNAME')
config['DB_PASSWORD'] = os.getenv('DB_PASSWORD')
config['DB_NAME'] = os.getenv('DB_NAME')
config['DB_PORT'] = os.getenv('DB_PORT')

##############################################################################
