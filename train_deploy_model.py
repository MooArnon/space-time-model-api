##########
# Import #
##############################################################################

import argparse
import subprocess

from config.config import config

#############
# Functions #
##############################################################################

def main(model_type: str) -> None:
    
    # Train model
    subprocess.run(
        [
            "python",
            "train_model.py",
            model_type,
        ]
    )

#########
# Usage #
##############################################################################

if __name__ == "__main__":
    # Parser to pass argument
    parser = argparse.ArgumentParser(
        description="Tag and push an image to a registry."
    )

    parser.add_argument(
        "model_type", 
        type=str, 
        help="The model type to be used for tagging the image."
    )
    
    # Run the code
    args = parser.parse_args()
    main(args.model_type)

##############################################################################
