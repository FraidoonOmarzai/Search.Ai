import os
from sklearn.feature_extraction import img_to_graph
from werkzeug.utils import secure_filename
from keras.utils import load_img,img_to_array
import tensorflow as tf
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19
from flask import Flask, url_for, render_template, redirect, request
import flask
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')


# Loading main page
@app.route('/')
def main():
    return render_template('main.html')


@app.route('/math')
def math():
    return render_template('math.html')


# SMS Detector
@app.route('/sms')
def smsDetector():
    return render_template('sms.html')


@app.route('/Spamprediction', methods=['POST'])
def Spamprediction():
    model = pickle.load(
        open('Trained Model/sms-detector/spam-model.pkl', 'rb'))
    tfv = pickle.load(
        open('Trained Model/sms-detector/CountVectorizer-transform.pkl', 'rb'))

    if request.method == 'POST':
        message = request.form["msg"]
        data = [message]
        msg = tfv.transform(data).toarray()
        result = model.predict(msg)

    if(int(result) == 1):
        prediction = "This is a SPAM message!"
    else:
        prediction = "This is NOT a spam message."
    return(render_template("result.html", prediction_text=prediction))


# Restaurant Review Sentiment Analysis
@app.route('/RestaurantReview')
def RestaurantR():
    return render_template('reviewR.html')


@app.route('/predictR', methods=['POST'])
def predictR():
    model = joblib.load('Trained Model/Restaurant Review/review-model.pkl')
    tfv = joblib.load('Trained Model/Restaurant Review/tfv-transform.pkl')

    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect = tfv.transform(data).toarray()
        result = model.predict(vect)

    if (int(result) == 1):
        prediction = "This is a POSITIVE Review"
    else:
        prediction = "This is a NEGATIVE Review"
    return render_template('result.html', prediction_text=prediction)


# VGG 19
# Load the saved Model
model = VGG19(weights='imagenet')


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/vgg19', methods=['GET'])
def vgg19():
    return render_template('vgg19.html')


@app.route('/predictImg', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])

        return result

    return None


# House Price Prediction
@app.route("/House")
def house_price():
    return render_template("house_price.html")


@app.route('/predictHP', methods=["POST"])
def predictHP():

    if request.method == "POST":

        Location = request.form['Location']
        Rooms = request.form['Rooms']
        Type = request.form['Type']
        Postcode = request.form['Postcode']
        Distance = request.form['Distance']
        Year = request.form['Year']

        input_variables = pd.DataFrame([[Location, Rooms, Type, Postcode, Distance, Year]],
                                       columns=['Suburb', 'Rooms', 'Type',
                                                'Postcode', 'Distance', 'Year'],
                                       dtype=float)

        model = joblib.load(
            "Trained Model/housePrice/housepriceprediction.joblib")
        prediction = model.predict(input_variables)[0]

        prediction = "Price of the house is: " + str(prediction) + "$"
        print(prediction)

    return(render_template("result.html", prediction_text=prediction))


if __name__ == '__main__':
    app.config['Debug'] = True
    app.run(host='0.0.0.0', port=8080)
