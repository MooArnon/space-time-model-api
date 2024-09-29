##########
# Import #
##############################################################################

import argparse
import glob
import logging 
import os
import time
import shutil

import pandas as pd
from scipy.stats import uniform, randint

from space_time_modeling.fe import ClassificationFE 
from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import ClassificationModel 
from space_time_modeling.modeling.__classification_wrapper import ClassifierWrapper
from space_time_modeling.utilities import load_instance
from space_time_pipeline.data_lake_house import Athena

logger = logging.getLogger(__name__)

###########
# Statics #
##############################################################################

n_window = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 25, 30,]
test_size = 0.005
n_iter = 2
feature_rank = 20

replace_condition_dict = {
    "<ASSET>" : "BTCUSDT",
    "<LIMIT>" : 50000
}

##############################################################################
# Param dict #
##############

xgboost_params_dict = {
    'learning_rate': uniform(0.01, 0.9),
    'n_estimators': randint(8, 512),
    'max_depth': randint(10, 200),
    'subsample': uniform(0.01, 0.9),
    'colsample_bytree': uniform(0.01, 0.9),
    'gamma': uniform(0, 0.9)
}

catboost_params_dict = {
    'iterations': randint(10, 200),
    'learning_rate': uniform(0.01, 1.0),
    'depth': randint(1, 16),
}

svc_params_dict = {
    'C': [0.01, 0.1, 1, 10, 50, 100],
    'gamma': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

random_forest_params_dict = {
    'n_estimators': randint(2, 200),  
    'max_features': ['log2', 'sqrt'],
    'max_depth': randint(2, 150), 
    'min_samples_split': randint(2, 100),  
    'min_samples_leaf': randint(1, 12), 
    'bootstrap': [True, False]  
}

logistic_regression_params_dict = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l1', 'l2'] 
}

knn_params_dict = {
    'n_neighbors': randint(1, 50),
    'weights': ['uniform', 'distance'],  
    'p': [1, 2] 
}

#########
# Train #
##############################################################################

def train_model(model_type: str, result_path: str = 'classifier') -> None:
    # statics 
    label_column = "signal"
    control_column = "scraped_timestamp"
    target_column = "price"
    
    # Feature col 
    feature_column = [
        "percent_change_df",
        "rsi_df",
        "date_hour_df",
        "ema",
        "percent_diff_ema",
    ]
    
    ununsed_feature = [f"ema_{win}" for win in n_window]
    
    df_path = os.path.join("btc-all.csv")
    
    # Preprocess data
    df = pd.read_csv(df_path)
    df.dropna(inplace=True)
    df = df[[target_column, control_column]]
    
    fe = ClassificationFE(
        control_column = control_column,
        target_column = target_column,
        label = label_column,
        fe_name_list = feature_column,
        n_window = n_window,
        ununsed_feature = ununsed_feature,
    )
    
    df_label = fe.add_label(
        df,
        target_column
    )
    
    df_train = fe.transform_df(
        df_label
    )
    
    # return df.columns
    # Train model
    modeling: ClassificationModel = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join(result_path),
        test_size = test_size,
        n_iter = n_iter,
        xgboost_params_dict = xgboost_params_dict,
        catboost_params_dict = catboost_params_dict,
        svc_params_dict = svc_params_dict,
        random_forest_params_dict = random_forest_params_dict,
        logistic_regression_params_dict = logistic_regression_params_dict,
        knn_params_dict = knn_params_dict,
        cv = 3,
        push_to_s3 = True,
        aws_s3_bucket = 'space-time-model',
        aws_s3_prefix = 'classifier/btc',
    )
    
    modeling.modeling(
        df = df_train, 
        preprocessing_pipeline=fe,
        model_name_list=[model_type],
        
        feature_rank = feature_rank,
    )
    
########
# Test #
##############################################################################

def test_model(path: str, type: str) -> None:
    model_path = os.path.join(
        path,
        type,
        f"{type}.pkl",
    )
    
    data_path = os.path.join(
        "local",
        "btc-all.csv",
    )
    
    # Load model
    model: ClassifierWrapper = load_instance(model_path)
    
    print(model.version)
    print(model.name)
    
    data_df = pd.read_csv(data_path)
    pred = model(data_df)
    
    print(pred)
    print('\n')
    
#############
# Utilities #
##############################################################################

def move_file(source_dir: str) -> None:
    """To move file from source dir to working directory

    Parameters
    ----------
    source_dir : str
        Source directory can be * as a suffix to filter classifier
        result; eg. '/path/to/prefix*'
    """
    # Get the current working directory
    dest_dir = os.getcwd()

    # Find all .pkl files in the source directory and its subdirectories
    pkl_files = glob.glob(os.path.join(source_dir, '**', '*.pkl'), recursive=True)

    # Move each .pkl file to the working directory
    for file in pkl_files:
        shutil.move(file, dest_dir)

    print(f"Moved {len(pkl_files)} .pkl files to {dest_dir}")

##############################################################################

def move_pkl_files(source_dir_prefix):
    """
    """
    # Get the current working directory
    dest_dir = os.getcwd()

    # Find directories that match the prefix
    source_dirs = glob.glob(source_dir_prefix)

    total_files_moved = 0

    # Loop through each directory that matches the prefix
    for source_dir in source_dirs:
        # Find all .pkl files in the source directory and its subdirectories
        pkl_files = glob.glob(
            os.path.join(source_dir, '**', '*.pkl'), 
            recursive=True
        )

        # Move each .pkl file to the working directory
        for file in pkl_files:
            shutil.move(file, dest_dir)
            total_files_moved += 1

    print(f"Moved {total_files_moved} .pkl files to {dest_dir}")

#######
# Use #
##############################################################################

if __name__ == "__main__":
    
    ##########################################################################
    # Parse arguments #
    ###################
    
    # Parser to pass argument
    parser = argparse.ArgumentParser(
        description="Tag and push an image to a registry."
    )

    parser.add_argument(
        "model_type", 
        type=str, 
        help="The model type to be used for tagging the image."
    )
    args = parser.parse_args()
    
    ##########################################################################
    # Select data #
    ###############
    
    # Export to csv for the next task
    lake_house = Athena(logger)
    data = lake_house.select(
        replace_condition_dict=replace_condition_dict,
        database="warehouse",
        query_file_path="framework/sql/select_data.sql",
    )
    df = pd.DataFrame(data)
    df.to_csv("btc-all.csv")
    
    ##########################################################################
    # Train model #
    ###############
    
    start_time = time.time()
    train_model(model_type=args.model_type)
    end_time = time.time()
    
    elapsed_time = (end_time - start_time)/60
    print(f"Elapsed time: {elapsed_time:.4f} mins")
    
    ##########################################################################
    # Move model out from result dir #
    ##################################
    
    move_pkl_files(source_dir_prefix="classifier*")

    ##########################################################################

##############################################################################
