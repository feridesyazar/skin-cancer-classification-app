import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

MODEL_CANDIDATES = [
    "skin_cancer_model.keras",
    "skin_cancer_model.h5",
    "skin_cancer.TL.h5",
]


def find_model_file() -> str:
    env_model = os.getenv("MODEL_PATH")
    if env_model and Path(env_model).exists():
        return env_model

    for model_file in MODEL_CANDIDATES:
        if Path(model_file).exists():
            return model_file

    raise FileNotFoundError(
        "No model file was found. Place skin_cancer_model.keras "
        "in the same directory as app.py."
    )


MODEL_PATH = find_model_file()
model = load_model(MODEL_PATH, compile=False)

input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

IMG_HEIGHT = int(input_shape[1] or 170)
IMG_WIDTH = int(input_shape[2] or 170)

# With the default alphabetical folder order used by Keras:
# Cancer = 0 and Non_Cancer = 1.
# Therefore, a sigmoid output represents the probability of Non-Cancer.
BINARY_CLASS_NAMES = ["Cancer", "Non-Cancer"]

MULTI_CLASS_NAMES_7 = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevus",
    "Vascular lesion",
]


def prepare_image(file_stream) -> np.ndarray:
    image = Image.open(file_stream).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(file_stream):
    image_array = prepare_image(file_stream)
    prediction = np.asarray(model.predict(image_array, verbose=0)).squeeze()

    # Binary sigmoid output.
    if prediction.ndim == 0:
        non_cancer_probability = float(np.clip(prediction, 0.0, 1.0))
        cancer_probability = 1.0 - non_cancer_probability

        probabilities = np.array(
            [cancer_probability, non_cancer_probability],
            dtype="float64",
        )
        best_index = int(np.argmax(probabilities))

        top_predictions = sorted(
            zip(BINARY_CLASS_NAMES, probabilities.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

        return (
            BINARY_CLASS_NAMES[best_index],
            float(probabilities[best_index]),
            top_predictions,
        )

    # Multi-class output.
    values = prediction.astype("float64")

    if (
        values.min() < 0
        or values.max() > 1
        or not np.isclose(values.sum(), 1.0, atol=1e-2)
    ):
        probabilities = softmax(values).numpy()
    else:
        probabilities = values / values.sum()

    output_size = len(probabilities)

    if output_size == 2:
        class_names = BINARY_CLASS_NAMES
    elif output_size == 7:
        class_names = MULTI_CLASS_NAMES_7
    else:
        class_names = [f"Class {index}" for index in range(output_size)]

    best_index = int(np.argmax(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        (class_names[index], float(probabilities[index]))
        for index in top_indices
    ]

    return (
        class_names[best_index],
        float(probabilities[best_index]),
        top_predictions,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_data = None
    top_predictions = []

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please select an image before starting the analysis."
        else:
            try:
                file_bytes = uploaded_file.read()

                mime_type = uploaded_file.mimetype or "image/jpeg"
                encoded_image = base64.b64encode(file_bytes).decode("utf-8")
                image_data = f"data:{mime_type};base64,{encoded_image}"

                label, confidence, top_predictions = predict_image(
                    io.BytesIO(file_bytes)
                )

                result = {
                    "label": label,
                    "confidence": confidence,
                    "is_cancer": label == "Cancer",
                }

            except (UnidentifiedImageError, OSError, ValueError):
                error = (
                    "The selected file could not be processed. "
                    "Please upload a valid JPG, JPEG, PNG, or WEBP image."
                )
            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "The image could not be analysed due to an application error. "
                    "Please try another image."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_data=image_data,
        top_predictions=top_predictions,
        model_path=Path(MODEL_PATH).name,
        image_size=f"{IMG_WIDTH} × {IMG_HEIGHT}",
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="The image is too large. Please upload a file smaller than 10 MB.",
        image_data=None,
        top_predictions=[],
        model_path=Path(MODEL_PATH).name,
        image_size=f"{IMG_WIDTH} × {IMG_HEIGHT}",
    ), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
