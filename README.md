Video Fruit Ripeness Detection App
==================================

A small, containerized app that detects fruit of different ripeness on the fly, applying light correction and adjustable a size filter to differentiate background/foreground objects. 


## Running in local machine

1. Run the shell script in Terminal to set up a virtual environment using Poetry:

```bash
source setup.sh
```

2. Execute the python script:

```bash
poetry run python app/app.py
```

3. Open the user interface at local host: [http://127.0.0.1:5000](http://127.0.0.1:5000)

    Use the buttons below the video stream to:
    - Toggle lighting condition (on: videos corrected to the reference image)
    - Toggle size filter (on: filtering out background fruits as specified in config)
    - Switch between two demo videos


## Running in docker environment

1. Build docker image

```bash
docker build -t video-fruit-ripeness-detection .
```

2. Run container (mount the target directory from host)

```bash
docker run -p 5000:5000 video-fruit-ripeness-detection
```

3. Open the user interface at local host: [http://127.0.0.1:5000](http://127.0.0.1:5000)

    Use the buttons below the video stream to:
    - Toggle lighting condition (on: videos corrected to the reference image)
    - Toggle size filter (on: filtering out background fruits as specified in config)
    - Switch between two demo videos

## Project structure

```bash

    .
    ├── README.md
    ├── setup.sh          # Shell script to set up a Poetry environment
    ├── poetry.lock      
    ├── pyproject.toml    # Poetry file that defines the python library dependencies
    ├── app              
    │   ├── config*       # Configuration files to specify the default setups
    │   ├── model.pt      # YOLOv8 object detection / instance segmentation model
    │   ├── frontend.html # Frontend to show the user interface
    │   ├── app.py        # Flask app to launch the detection service
    │   └── main.py       # Source script to run the inference generator
    └── data              
        ├── *.MP4         # Example video files to make inference from
        └── reference.jpg # reference image for lighting correction
```