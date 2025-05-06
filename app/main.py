import cv2
from PIL import Image
from skimage.exposure import match_histograms

import argparse
import numpy as np

from ultralytics import YOLO
import supervision as sv
import timeit

SIZE_FILTER = True
CORRECT_LIGHTING = True
FOREGROUND_FRUIT_SIZE = 50000
BACKGROUND_FRUIT_SIZE = 7000
CONF_THRESHOLD = 20
REFERENCE = "reference.jpg"

# Scenario 1: day light - background size = 7000; foreground: 50000
# video_path = "day_light.mp4"

# Scenario 2: pink light - background size = 3500; foreground: 30000
# video_path = "pink_light.mp4"

# Scenario 3: under shielded screen - background size = 7000; foreground: 50000
video_path = "shielded_screen.mp4"

model_path = "model.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1080, 1920], nargs=2, type=int)
    args = parser.parse_args()
    return args


def correct_for_lighting(frame):
    """use histogram matching to correct for lighting"""
    img_ref = Image.open(REFERENCE).convert("RGB")
    img_ref = np.array(img_ref)
    frame = match_histograms(frame, img_ref, channel_axis=-1)
    return frame


def inference_over_inhouse(results):
    """reformat inference result from YOLOv8 inhouse model"""

    image = {}
    image["width"] = results.orig_shape[1]
    image["height"] = results.orig_shape[0]

    predictions = []
    boxes = results.boxes.xywh.cpu()
    for i, item in enumerate(results.summary()):
        prediction = {}
        prediction["x"] = boxes[i][0].numpy().item()  # x (center coordinate)
        prediction["y"] = boxes[i][1].numpy().item()  # y (center coordinate)
        prediction["width"] = boxes[i][2].numpy().item()
        prediction["height"] = boxes[i][3].numpy().item()
        prediction["confidence"] = item["confidence"]
        prediction["class"] = item["name"]
        points = []
        for x, y in zip(item["segments"]["x"], item["segments"]["y"]):
            temp = {"x": x, "y": y}
            points.append(temp)
        prediction["points"] = points
        prediction["class_id"] = item["class"]
        predictions.append(prediction)

    result_dict = {"image": image, "predictions": predictions}

    return result_dict


def remove_prefix(s, prefix):
    return s[len(prefix) :] if s.startswith(prefix) else s


def merge_class_ids(detections):
    '''deal with prefixes e.g. "b_" & "l_"'''
    # result.names
    id = detections.class_id
    new_id = []
    for i in id:
        if i == 3:
            i = 0
        elif i == 4:
            i = 1
        elif i == 5:
            i = 2
        new_id.append(i)
    new_id = np.array(new_id)
    detections.class_id = new_id
    return detections


def merge_labels(model, detections):
    '''deal with prefixes e.g. "b_" & "l_"'''
    labels = [
        f"{model.model.names[class_id]}"  # {confidence:0.2f}
        for _, _, confidence, class_id, _, _ in detections
    ]
    new_labels = [remove_prefix(label, "b_") for label in labels]
    labels = [remove_prefix(label, "l_") for label in new_labels]
    return labels


def filter_by_size(
    label_ls: list,
    pixel_size_threshold: int = BACKGROUND_FRUIT_SIZE,
):
    """
    Apply size filter to capture foreground instances.
    """
    bbox_fruits = [
        x for x in label_ls if x["height"] * x["width"] < FOREGROUND_FRUIT_SIZE
    ]  # hardcoded fixed threshold for fruits
    sizes = [x["height"] * x["width"] for x in bbox_fruits]
    bbox_background = [
        x for x in bbox_fruits if x["height"] * x["width"] < pixel_size_threshold
    ]
    bbox_foreground = [x for x in bbox_fruits if x not in bbox_background]
    return bbox_background, bbox_foreground, sizes


def apply_filter_sinlge_image(result, pixel_size_threshold: int = BACKGROUND_FRUIT_SIZE):
    """Apply foreground filter and plot on image"""
    label_ls = result["predictions"]
    _, bbox_foreground, sizes = filter_by_size(
        label_ls,
        pixel_size_threshold=pixel_size_threshold,
    )
    filtered_result = {
        "predictions": bbox_foreground,
        "image": result["image"],
    }
    return filtered_result, bbox_foreground, sizes


def main(
    model_path: str,
    video_path: str,
    correct_lighting: bool = True,
    size_filter: bool = True,
):
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(video_path)  # 0 for Webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO(model_path)

    label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2, text_padding=2)
    mask_annotator = sv.BoundingBoxAnnotator()

    while True:
        ret, frame = cap.read()

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # pre-processing
        if correct_lighting:
            starttime = timeit.default_timer()
            frame = correct_for_lighting(frame)
            print(
                "* correct lighting takes:",
                (round(timeit.default_timer() - starttime, ndigits=2)) * 1000,
                "ms",
            )

        # make inference
        result = model.predict(frame, save_conf=True, conf=CONF_THRESHOLD/100, device="cpu")[0]

        # filter background fruit
        if size_filter:
            starttime = timeit.default_timer()
            result_dict = inference_over_inhouse(result)
            filtered_result, _, _ = apply_filter_sinlge_image(result_dict)
            detections = sv.Detections.from_inference(filtered_result)
            print(
                "* filtering background takes:",
                (round(timeit.default_timer() - starttime, ndigits=4)) * 1000,
                "ms\n ",
            )
        else:
            detections = sv.Detections.from_ultralytics(result) # if no size filter

        # annotate detections
        detections = merge_class_ids(detections) 
        labels = merge_labels(model, detections)
        annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

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

    # Release video sources
    cap.release()


if __name__ == "__main__":
    main(
        model_path,
        video_path,
        size_filter=SIZE_FILTER,
        correct_lighting=CORRECT_LIGHTING,
    )
