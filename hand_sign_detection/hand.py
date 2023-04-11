from __future__ import division, print_function
import os
import numpy as np
import cv2 as cv

# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from cvzone.HandTrackingModule import HandDetector

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model\hand_detect_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')
# detector =HandDetector(detectionCon=0.5, maxHands=1)
classes = np.load('classes.npy',allow_pickle=True)
print(classes)

def model_predict(img_path, model):
    input=[]
    # ig = image.load_img(img_path, target_size=(224, 224))
    print("############",img_path)
    img_array = cv.imread(img_path)
    img=img_array.copy()
    # hands, immg2 = detector.findHands(img_array)
    # if len(hands)==1:
    #     d=hands[0]['bbox']
    #     (x,y,w,h)=d
    #     hand_roi = img[y:y+h, x:x+w]
    resized = cv.resize(img, (224,224), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 7,9)
    # gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    # adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)
    input.append(canny)
    predcit= model.predict(np.array(input))
    # print(f"prediction is: {p[np.argmax(predcit)]}")
    return classes[np.argmax(predcit)]
    # else:
    #     print("no hand detected try again")
    #     return "no hand detected try again" 
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x, mode='caffe')
    # preds = model.predict(x)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        img_pth="./assets/"+f.filename
        f.save(img_pth)
        preds = model_predict(img_pth, model)

        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)          # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)