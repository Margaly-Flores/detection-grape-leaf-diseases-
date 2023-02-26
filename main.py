from flask import Flask, render_template, request
from inference import ObjectDetection
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def home():
    return render_template("start.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    errors = []
    allowed_extensions = set(["png", "jpg", "jpeg", "gif"])
    try:
        if request.method == "POST":
            currentfile = request.files.get("file", "")
            # prediction of model
        model = ObjectDetection()
        prediction, classes = model(currentfile)

        buffered = BytesIO()
        image_result = Image.fromarray(prediction)
        image_result = image_result.resize((400, 400))
        image_result.save(buffered, format="JPEG")
        image_memory = base64.b64encode(buffered.getvalue())
        print("classes", classes)
        return render_template(
            "result.html", classes=", ".join(classes), img_data=image_memory.decode("utf-8")
        )

    except Exception as e:
        return {"response": f"Ocurrieron los siguientes errores: {e}"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=os.getenv("PORT", default=5000))
