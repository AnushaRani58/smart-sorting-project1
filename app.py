import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("fruit_model.h5")

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names (same order as training folders)
class_names = [
    'FreshApple',
    'FreshBanana',
    'FreshStrawberry',
    'RottenApple',
    'RottenBanana',
    'RottenStrawberry'
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return class_names[class_index], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename)
# 


if __name__ == "__main__":
    app.run(debug=True)