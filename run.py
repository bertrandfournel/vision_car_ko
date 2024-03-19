import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import json

from keras.models import load_model

import base64
from io import BytesIO

# import monitor and logging for debugging
import flask_monitoringdashboard as dashboard
import logging

# set the logging output file and incident level to Debug(incident level 10) or Info (level 20)
logging.basicConfig(filename='app.log' ,level=logging.INFO)

app = Flask(__name__)
# integrate the dashboard to the app
dashboard.bind(app)

from jinja2.exceptions import TemplateNotFound

@app.route('/')
def index():
    try:
        app.logger.info('User access to Index page')
        return render_template('index.html')
    except TemplateNotFound as e:
        app.logger.error('Error: %s', e)
        raise e

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    try:
        app.logger.info('Upload image route')
        model = load_model("unet_vgg16_categorical_crossentropy_raw_data.keras", compile=False)


        colors = np.array([[ 68,   1,  84],
        [ 70,  49, 126],
        [ 54,  91, 140],
        [ 39, 126, 142],
        [ 31, 161, 135],
        [ 73, 193, 109],
        [159, 217,  56],
        [253, 231,  36]])
        

        if request.method == 'POST':

            image = request.files['file']

        if image.filename == '':
            return "Nom de fichier invalide"

            img = Image.open(image)

            IMAGE_SIZE = 512

            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
            img_resized = np.array(img_resized)

            img_resized = np.expand_dims(img_resized, 0)
            img_resized = img_resized / 255.

            predict_mask = model.predict(img_resized, verbose=0)
            predict_mask = np.argmax(predict_mask, axis=3)
            predict_mask = np.squeeze(predict_mask, axis=0)
            predict_mask = predict_mask.astype(np.uint8)
            predict_mask = Image.fromarray(predict_mask)
            predict_mask = predict_mask.resize((img.size[0], img.size[1]), resample=Image.Resampling.NEAREST)
            
            predict_mask = np.array(predict_mask)
            predict_mask = colors[predict_mask]
            predict_mask = predict_mask.astype(np.uint8)

            buffered_img = BytesIO()
            img.save(buffered_img, format="PNG")
            base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

            buffered_mask = BytesIO()
            base64_mask = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

            print("Finished")

            return json({'message':"predict ok", "img_data":base64_img, "mask_data":base64_mask})
    except Exception as e:
        app.logger.error('Eroor: %s', e)
        return jsonify({'message': 'An error occurred: {}'.format(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)