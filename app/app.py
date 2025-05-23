import logging

from config import get_video_path, get_video_settings, model_options
from flask import Flask, Response, jsonify, render_template, request
from main import inference_generator

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder=".")

# Global variables to store the currently selected video and its settings
current_video = get_video_path("tomato_1.mov")  # Default video path
video_settings = get_video_settings("tomato_1.mov")  # Default video settings


@app.route("/")
def index():
    return render_template("frontend.html")


@app.route("/video_feed")
def video_feed():
    try:
        response = Response(
            inference_generator(current_video, video_settings),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        app.logger.error(f"Error in video_feed: {e}")
        return "Error in video feed", 500


@app.route("/toggle_option", methods=["POST"])
def toggle_option():
    data = request.get_json()
    option = data.get("option")
    if option in model_options:
        model_options[option] = not model_options[option]
        app.logger.info(f"{option.capitalize()} toggled to {model_options[option]}")
        return jsonify(
            {"message": f"{option.capitalize()} toggled to {model_options[option]}."}
        )
    else:
        return jsonify({"message": "Invalid option."}), 400


@app.route("/switch_video", methods=["POST"])
def switch_video():
    global current_video, video_settings
    data = request.get_json()
    video = data.get("video")
    if video:
        current_video = get_video_path(video)
        video_settings = get_video_settings(video)
        app.logger.info(f"Switched to video: {current_video}")
        app.logger.info(f"Updated video settings: {video_settings}")
        return jsonify({"message": f"Switched to video: {current_video}"})
    else:
        return jsonify({"message": "Invalid video selection."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
