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

SCENARIO: str = config["selected_video"]
FOREGROUND_FRUIT_SIZE: int = config["experimentation"][SCENARIO][
    "foreground_fruit_size"
]
BACKGROUND_FRUIT_SIZE: int = config["experimentation"][SCENARIO][
    "background_fruit_size"
]
RESOLUTION: list = config["experimentation"][SCENARIO]["video_resolution"]

DATA_DIR: str = config["data_dir"]
REFERENCE: str = config["reference_path"]
VIDEO_PATH: str = f"{DATA_DIR}/{SCENARIO}"
MODEL_PATH: str = config["model_path"]
