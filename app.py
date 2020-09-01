import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
import cv2

app = Flask(__name__)


model = load_model("./models/pn_model.h5")
# model = load_model("my_model.h5")


def model_predict(img, model):
    x = cv2.imread("./uploads1/image.png", cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (150,150))
    x = np.array(x)
    x = x / 255
    x = x.reshape(-1,150,150,1)
    preds = model.predict(x)
    if preds[0]>0.5:
        result = f'Normal with {round(preds[0][0]*100, 2)}% probability'
    else:
        result = f'Pneumonia with {round((1-preds[0][0])*100, 2)}% probability'
    return result
    # return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img.save("./uploads1/image.png")
        return jsonify(result=model_predict(img, model))
    return None

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run()
