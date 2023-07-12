from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('symptoms.pkl', 'rb'))

app = Flask(__name__)

@app.route('/disease_predictor')
def disease_predictor():
    return render_template('disease_predictor.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    disease = model.predict(arr)[0]
    result = {'result': disease}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)