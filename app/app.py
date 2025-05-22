import logging

from flask import Flask, Response, render_template
from main import main_inference_generator

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder=".")


@app.route("/")
def index():
    return render_template("frontend.html")


@app.route("/video_feed")
def video_feed():
    try:
        return Response(
            main_inference_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        app.logger.error(f"Error in video_feed: {e}")
        return "Error in video feed", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
