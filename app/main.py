import logging
import timeit

import cv2
import numpy as np
import supervision as sv
from config import (
    CONF_THRESHOLD,
    INHOUSE_MODEL,
    MODEL_API_KEY,
    MODEL_API_URL,
    MODEL_PATH,
    MODEL_PROJECT_NAME,
    MODEL_VERSION,
    REFERENCE,
    model_options,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image
from skimage.exposure import match_histograms
from ultralytics import YOLO


def correct_for_lighting(frame: np.array) -> np.array:
    """Use histogram matching to correct for lighting."""
    try:
        img_ref: Image = Image.open(REFERENCE)
        img_ref: np.array = np.array(img_ref)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        frame: np.array = match_histograms(frame, img_ref, channel_axis=-1)
        return frame
    except Exception as e:
        logging.error(f"Error in correct_for_lighting: {e}")
        return frame  # Return the original frame if correction fails


def reformat_inference(results: dict) -> dict:
    """Reformat inference result from YOLOv8 model."""

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


def remove_prefix(s: str, prefix: str) -> str:
    """Remove prefix from string."""
    return s[len(prefix) :] if s.startswith(prefix) else s


def merge_class_ids(detections: dict) -> dict:
    """Merge class IDs to remove prefixes."""
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


def merge_labels(labels: list) -> list:
    """Merge labels to remove prefixes."""
    new_labels = [remove_prefix(label, "b_") for label in labels]
    labels = [remove_prefix(label, "l_") for label in new_labels]
    return labels


def filter_by_size(
    label_ls: list,
    background_size_threshold: int,
    foreground_size_threshold: int,
) -> tuple[list, list, list]:
    """Apply size filter to capture foreground instances."""
    bbox_fruits = [
        x for x in label_ls if x["height"] * x["width"] < foreground_size_threshold
    ]
    sizes = [x["height"] * x["width"] for x in bbox_fruits]
    bbox_background = [
        x for x in bbox_fruits if x["height"] * x["width"] < background_size_threshold
    ]
    bbox_foreground = [x for x in bbox_fruits if x not in bbox_background]
    return bbox_background, bbox_foreground, sizes


def apply_filter_sinlge_image(
    result: dict, background_size_threshold: int, foreground_size_threshold: int
):
    """Apply foreground filter and plot on image."""
    try:
        label_ls = result["predictions"]
        _, bbox_foreground, sizes = filter_by_size(
            label_ls,
            background_size_threshold=background_size_threshold,
            foreground_size_threshold=foreground_size_threshold,
        )
        logging.info(f"Filtered {len(bbox_foreground)} foreground objects.")
        filtered_result = {
            "predictions": bbox_foreground,
            "image": result["image"],
        }
        return filtered_result, bbox_foreground, sizes
    except Exception as e:
        logging.error(f"Error in apply_filter_sinlge_image: {e}")
        return result, [], []


def inference_over_api(frame: Image) -> dict:
    """Configure and predict using a pre-trained roboflow model over API.

    Args:
        image: PIL images, NumPy arrays, URLs, or filenames.

    Returns:
        Predictions from a roboflow pre-trained model.

    """
    custom_configuration = InferenceConfiguration(
        confidence_threshold=CONF_THRESHOLD / 100
    )
    inf_client = InferenceHTTPClient(
        api_url=MODEL_API_URL,
        api_key=MODEL_API_KEY,
    )
    with inf_client.use_configuration(custom_configuration):
        result = inf_client.infer(
            frame, model_id=f"{MODEL_PROJECT_NAME}/{MODEL_VERSION}"
        )
    return result


def inference_generator(current_video, video_settings, class_counts_callback=None):
    """Main inference generator function."""

    correct_lighting = model_options["lighting"]
    size_filter = model_options["filtering"]
    logging.info(
        f"Lighting correction: {correct_lighting}, Size filtering: {size_filter}"
    )
    logging.info(f"Using video settings: {video_settings}")

    # Initialize video capture with dynamic video path and settings
    frame_width, frame_height = video_settings["video_resolution"]
    cap = cv2.VideoCapture(current_video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if INHOUSE_MODEL:
        logging.info("Making inference using a self-trained model.")
        model = YOLO(MODEL_PATH)
    else:
        logging.info("Making inference using a roboflow model.")

    label_annotator = sv.LabelAnnotator(
        text_scale=0.6, text_thickness=1, text_padding=4
    )
    mask_annotator = sv.BoundingBoxAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dynamically check the current state of model_options
        correct_lighting = model_options["lighting"]
        size_filter = model_options["filtering"]

        # Pre-processing
        if correct_lighting:
            starttime = timeit.default_timer()
            frame: np.array = correct_for_lighting(frame)
            print(
                "* correct lighting takes:",
                (round(timeit.default_timer() - starttime, ndigits=2)) * 1000,
                "ms",
            )

        # Make inference
        if INHOUSE_MODEL:
            result = model.predict(
                frame, save_conf=True, conf=CONF_THRESHOLD / 100, device="cpu"
            )[0]
        else:
            result = inference_over_api(frame)

        # Filter background fruit
        if size_filter:
            starttime = timeit.default_timer()
            if INHOUSE_MODEL:  # remformat to make compatible with roboflow inference
                result_dict: dict = reformat_inference(result)
            else:
                result_dict: dict = result
            filtered_result, _, _ = apply_filter_sinlge_image(
                result_dict,
                background_size_threshold=video_settings["background_fruit_size"],
                foreground_size_threshold=video_settings["foreground_fruit_size"],
            )
            detections = sv.Detections.from_inference(filtered_result)
            print(
                "* filtering background takes:",
                (round(timeit.default_timer() - starttime, ndigits=4)) * 1000,
                "ms\n ",
            )
        else:
            if INHOUSE_MODEL:
                detections = sv.Detections.from_ultralytics(result)
            else:
                detections = sv.Detections.from_inference(result)

        # Count by class
        detections: dict = merge_class_ids(detections)
        if INHOUSE_MODEL:
            labels = [
                f"{model.model.names[class_id]}"
                for _, _, confidence, class_id, _, _ in detections
            ]
        else:
            labels = detections.data["class_name"].tolist()
        labels: list = merge_labels(labels)
        class_counts = np.unique(labels, return_counts=True)
        counts_dict = {str(k): int(v) for k, v in zip(*class_counts)}
        logging.info(f"Class counts: {counts_dict}")
        if class_counts_callback:
            class_counts_callback(counts_dict)

        # Annotate detections
        annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        # Generator that yields JPEG frames
        ret, buffer = cv2.imencode(".jpg", annotated_image)
        frame = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
