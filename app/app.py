import logging

from config import model_options
from flask import Flask, Response, jsonify, render_template, request
from main import inference_generator

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder=".")


@app.route("/")
def index():
    return render_template("frontend.html")


@app.route("/video_feed")
def video_feed():
    try:
        return Response(
            inference_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
