from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('symptoms.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main(): 
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    dis = model.predict(arr)[0]
    result = {'result': dis}
    # return the result as a JSON object
    return jsonify(result)
    # return render_template('prediction.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)