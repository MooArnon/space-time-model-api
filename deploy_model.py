##########
# Import #
##############################################################################

import argparse

from config.config import config
from utils.registry import tag_image, push_to_registry, build_image

##############
# Frunctions #
##############################################################################

def main(registry_endpoint: str, model_type: str) -> None:
    
    # Build
    build_image(
        model_type=model_type,
        docker_file_name=config["MODEL_TYPE_MAPPING"][model_type]
    )
    
    # Tag
    tag_image(
        registry_endpoint = registry_endpoint,
        model_type = model_type,
    )
    
    # Push
    push_to_registry(
        registry_endpoint = registry_endpoint,
        model_type = model_type,
    )

#########
# Usage #
##############################################################################

if __name__ == "__main__":
    # Parser to pass aguement
    parser = argparse.ArgumentParser(
        description="Tag and push an image to a registry."
    )
    parser.add_argument(
        "registry_endpoint", 
        type=str, 
        help="The registry endpoint URL."
    )
    parser.add_argument(
        "model_type", 
        type=str, 
        help="The model type to be used for tagging the image."
    )
    
    # Run the code
    args = parser.parse_args()
    main(args.registry_endpoint, args.model_type)

##############################################################################
