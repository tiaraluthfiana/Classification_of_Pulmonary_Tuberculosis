import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import scipy.stats as stats
import numpy as np
import os

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu, True
        )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOAD MODEL (.keras)
model = load_model("final_model.keras", compile=False)

IMG_SIZE = 512

# CLASS LABEL
classes = ['Normal', 'Tuberculosis']

# PREPROCESS IMAGE
def preprocess_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ROUTE
@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None
    confidence = None
    image_path = None
    probabilities = None

    if request.method == 'POST':

        file = request.files['file']

        if file:
            os.makedirs("static/uploads", exist_ok=True)

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(filepath)

            # preprocessing
            image = preprocess_image(filepath)

            # inference
            pred = model.predict(image, verbose=0)

            score = float(pred[0][0])

            # Threshold terbaik hasil tuning
            threshold = 0.569

            if score >= threshold:
                prediction = "Tuberculosis"
                confidence = score * 100
            else:
                prediction = "Normal"
                confidence = (1 - score) * 100

            probabilities = {
                "Normal": (1 - score) * 100,
                "Tuberculosis": score * 100
            }

            print(f"Score: {score:.4f}")

            image_path = 'uploads/' + file.filename

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        probabilities=probabilities
    )

if __name__ == '__main__':
    app.run(debug=True)