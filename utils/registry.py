##########
# Import #
##############################################################################

import subprocess

#############
# Functions #
##############################################################################

def build_image(
        model_type: str, 
        docker_file_name: str, 
        dockerfile_path: str='.'
) -> None:
    build_cmd = [
        "sudo",
        "docker", 
        "build", 
        "--platform=linux/amd64",
        "-f",
        f"{docker_file_name}.Dockerfile",
        "-t", 
        model_type, 
        "--build-arg", 
        f"MODEL_TYPE={model_type}", 
        dockerfile_path,
        "--no-cache"
    ]
    try:
        print(f"Building Docker image {model_type} for model type {model_type}...")
        result = subprocess.run(
            build_cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"Docker image {model_type} built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e.stderr}")
        raise

##############################################################################

def tag_image(
        registry_endpoint: str,
        model_type: str,
        tag: str = "latest",
    ) -> None:
    """Tag image

    Parameters
    ----------
    registry_endpoint : str
        End point of registry
    model_type : str
        Type of model, eg. xboost
    tag : str, optional
        tag of image, by default "latest"
    """
    # Command
    command = [
        "docker", 
        "tag", 
        f"{model_type}", 
        f"{registry_endpoint}/{model_type}:{tag}"
    ]
    
    # Run the command using subprocess
    process = subprocess.run(command, capture_output=True, text=True)

    # Print the output or error message
    if process.returncode == 0:
        print(f"Successfully tagged {model_type} image.")
    else:
        print(f"Error tagging image: {process.stderr}")
    
##############################################################################

def push_to_registry(
        registry_endpoint: str,
        model_type: str,
        tag: str = "latest",
    ) -> None:
    """Push image to target endpoint

    Parameters
    ----------
    registry_endpoint : str
        End point of registry
    model_type : str
        Type of model, eg. xboost
    tag : str, optional
        tag of image, by default "latest"
    """
    # Command
    command = [
        "docker", 
        "push",
        f"{registry_endpoint}/{model_type}:{tag}"
    ]
    
    # Run the command using subprocess
    process = subprocess.run(command, capture_output=True, text=True)

    # Print the output or error message
    if process.returncode == 0:
        print(f"Successfully pushed {model_type} image.")
    else:
        print(f"Error pushing image: {process.stderr}")

##############################################################################
