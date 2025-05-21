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
cd app
poetry run python main.py
```

### To visualize detection processing 

Uncomment the following lines (199-209) in **main.py**:

```python
# disable for docker container
cv2.imshow("yolov8", frame)
k = cv2.waitKey(1)
if k==27:    # Esc key to stop
    break
elif k==-1:  # normally -1 returned,so don't print it
    continue
else:
    print(k) # else print its value
# Destroy all OpenCV windows
cv2.destroyAllWindows() 
```

## Running in docker environment

1. Build docker image

```bash
docker build -t streaming .
```

2. Run container (mount the target directory from host)

```bash
docker run --rm --volume=$LOCAL_PROJECT_DIRECTORY/app/:/app -i streaming
```

### Make sure to turn off visualization 

Comment out the following lines (201-221) in **main.py**:

```python
# # disable for docker container
# cv2.imshow("yolov8", frame)
# k = cv2.waitKey(1)
# if k==27:    # Esc key to stop
#     break
# elif k==-1:  # normally -1 returned,so don't print it
#     continue
# else:
#     print(k) # else print its value
# # Destroy all OpenCV windows
# cv2.destroyAllWindows() 
```


## Project structure

```bash
    .
    ├── README.md
    ├── setup.sh          # Shell script to set up a Poetry environment
    ├── poetry.lock      
    ├── pyproject.toml    # Poetry file that defines the python library dependencies
    └── app              
        ├── *.MP4         # Example video files to make inference from
        ├── *.pt          # YOLOv8 object detection / instance segmentation model
        ├── main.py
        └── reference.jpg # reference image for lighting correction
```