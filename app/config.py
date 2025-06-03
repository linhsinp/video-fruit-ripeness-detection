import os

import yaml

with open("app/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SIZE_FILTER: bool = config["size_filter"]
CORRECT_LIGHTING: bool = config["correct_for_lighting"]

model_options = {
    "lighting": CORRECT_LIGHTING,
    "filtering": SIZE_FILTER,
}

CONF_THRESHOLD: int = config["config_threshold"]
COMPRESSED_DIR: str = config["compressed_dir"]
REFERENCE: str = config["reference_path"]

INHOUSE_MODEL: bool = config["inhouse_model"]
MODEL_PATH: str = config["model_path"]

MODEL_PROJECT_NAME: str = config["model_project_name"]
MODEL_API_URL: str = config["model_api_url"]
MODEL_API_KEY: str = config["model_api_key"]
MODEL_VERSION: int = config["model_version"]


def get_video_path(video_name: str) -> str:
    """Function to get the video path."""
    return os.path.join(COMPRESSED_DIR, video_name)


def get_video_settings(video_name_or_path: str):
    """Function to get video-specific settings."""
    video_name = os.path.basename(video_name_or_path)
    FOREGROUND_FRUIT_SIZE: int = config["experimentation"][video_name][
        "foreground_fruit_size"
    ]
    BACKGROUND_FRUIT_SIZE: int = config["experimentation"][video_name][
        "background_fruit_size"
    ]
    RESOLUTION: list = config["experimentation"][video_name]["video_resolution"]
    return {
        "foreground_fruit_size": FOREGROUND_FRUIT_SIZE,
        "background_fruit_size": BACKGROUND_FRUIT_SIZE,
        "video_resolution": RESOLUTION,
    }
