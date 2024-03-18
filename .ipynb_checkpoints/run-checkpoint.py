from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import io
import base64
import tensorflow as tf
import cv2
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():

    model = load_model("models/unet_light")

    print(model)

    colors = [[ 68,   1,  84, 255],
       [ 70,  49, 126, 255],
       [ 54,  91, 140, 255],
       [ 39, 126, 142, 255],
       [ 31, 161, 135, 255],
       [ 73, 193, 109, 255],
       [159, 217,  56, 255],
       [253, 231,  36, 255]]


    if request.method == 'POST':
        if 'image' not in request.files:
            return "Aucune image incluse dans la requête"

        image = request.files['image']
        print(type(image))

        if image.filename == '':
            return "Nom de fichier invalide"

        # Charger l'image avec PIL (Pillow)
        img = Image.open(image)
        print(type(img))
        

        # Convertir l'image en tableau NumPy
        img_array = np.array(img)
        print(type(img_array))
        print(img_array.shape)

        IMAGE_SIZE = 512

        img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
        img_resized = np.expand_dims(img_resized, 0)
        img_resized = img_resized.astype(np.float32)

        predict_mask = model.predict(img_resized, verbose=0)
        predict_mask = np.argmax(predict_mask, axis=3)
        predict_mask = np.squeeze(predict_mask, axis=0)
        predict_mask = predict_mask.astype(np.uint8)
        predict_mask = cv2.resize(predict_mask, (img_array.shape[1], img_array.shape[0]))

        print(predict_mask.shape)

        mask_serialisable = predict_mask.tolist()


        # # Effectuer des traitements sur l'image
        # # Exemple : Inverser les couleurs (négatif)
        # img_array_processed = 255 - img_array

        # # Convertir le résultat en format d'image
        # img_processed = Image.fromarray(img_array_processed.astype('uint8'))

        # rawBytes = io.BytesIO()
        # img_processed.save(rawBytes, "PNG")
        # rawBytes.seek(0)  # return to the start of the file
        # img_processed_data = base64.b64encode(rawBytes.read())
        # img_processed_data = img_processed_data.decode('UTF-8')

        # rawBytes = io.BytesIO()
        # img.save(rawBytes, "PNG")
        # rawBytes.seek(0)  # return to the start of the file
        # img_original_data = base64.b64encode(rawBytes.read())
        # img_original_data = img_original_data.decode('UTF-8')

        # # Convertir les images en base64 pour affichage
        # img_original_data = base64.b64encode(img_array.tobytes()).decode('utf-8')
        # img_processed_data = base64.b64encode(img_processed.tobytes()).decode('utf-8')

        print("Finished")

        return jsonify({'message':"predict ok", "image_array":mask_serialisable})

        #return jsonify({"message": "Predict done", "img_original": f"data:image/png;base64,{img_original_data}", "img_processed": f"data:image/png;base64,{img_processed_data}"})

    #return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)