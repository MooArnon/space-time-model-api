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
