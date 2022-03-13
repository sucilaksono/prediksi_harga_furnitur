from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
filename = 'Model_IKEA_XGB.sav'
loaded_model = pickle.load(open(filename, 'rb'))
cols = ['category', 'other_colors', 'num_designer', 'depth', 'height', 'width']

@app.route('/')
def home():
    return render_template("price.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = loaded_model.predict(data_unseen)
    prediction = int(prediction[0])
    return render_template('price.html',pred='perkiraan harga menjadi {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = loaded_model.predict(data_unseen)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
