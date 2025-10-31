import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"


model = load_model("mnist_cnn_model.keras")

def preprocess_image(image_path):
    """ Tiền xử lý ảnh: chuyển về grayscale, resize, chuẩn hóa """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (28, 28))  
    img = img.astype("float32") / 255.0 
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0) 
    return img

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file được tải lên"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file!"}), 400
    

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    

    processed_img = preprocess_image(filepath)
    prediction = model.predict(processed_img)
    predicted_label = int(np.argmax(prediction))  
    
    return jsonify({"digit": predicted_label})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  
    app.run(debug=True)
