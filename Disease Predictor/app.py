from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
# from flask_cors import CORS

model = pickle.load(open('pickles/symptoms.pkl', 'rb'))
model_diab = pickle.load(open('pickles/diab.pkl', 'rb'))
model_cvd = pickle.load(open('pickles/cvd.pkl', 'rb'))
model_chr_kid = pickle.load(open('pickles/chr_kid_dis.pkl', 'rb'))
model_stroke = pickle.load(open('pickles/stroke.pkl', 'rb'))
model_thyroid = pickle.load(open('pickles/thyroid.pkl', 'rb'))
model_parkinsons = pickle.load(open('pickles/parkinsons.pkl', 'rb'))
model_lung = pickle.load(open('pickles/lung_cancer.pkl', 'rb'))
model_cerv = pickle.load(open('pickles/cerv_canc.pkl', 'rb'))
model_age_chr_kid = pickle.load(open('pickles/age_chr_kid_dis.pkl', 'rb'))
model_age_cvd = pickle.load(open('pickles/age_cvd.pkl', 'rb'))
model_age_diab = pickle.load(open('pickles/age_diab.pkl', 'rb'))
model_age_stroke = pickle.load(open('pickles/age_stroke.pkl', 'rb'))
model_age_thyroid = pickle.load(open('pickles/age_thyroid.pkl', 'rb'))
model_age_lung = pickle.load(open('pickles/age_lung_cancer.pkl', 'rb'))
model_age_cerv = pickle.load(open('pickles/age_cerv_canc.pkl', 'rb'))

app = Flask(__name__)
# CORS(app)

@app.route('/disease_predictor')
def disease_predictor():
    return render_template('disease_predictor.html')

@app.route('/x_ray_predictor')
def x_ray_predictor():
    return render_template('x_ray_predictor.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    disease = model.predict(arr)[0]
    result = {'result': disease}
    return jsonify(result)

@app.route('/x_ray', methods=['POST'])
def x_ray():
    result = {'result': 7}
    return jsonify(result)

@app.route('/diab',methods=['POST'])
def diab():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_diab.predict(arr)[0]
    accuracy_diab=81
    result= {'result': int(num),'acc':accuracy_diab}
    return jsonify(result)


@app.route('/cvd',methods=['POST'])
def cvd():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_cvd.predict(arr)[0]
    accuracy_cvd=74.7
    result= {'result': int(num),'acc':accuracy_cvd}
    return jsonify(result)
    

@app.route('/chr_kid',methods=['POST'])
def chr_kid():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_chr_kid.predict(arr)[0]
    accuracy_chr_kid=97.5
    result= {'result': int(num),'acc':accuracy_chr_kid}
    return jsonify(result)


@app.route('/stroke',methods=['POST'])
def stroke():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_stroke.predict(arr)[0]
    accuracy_stroke=95.6
    result= {'result': int(num),'acc':accuracy_stroke}
    return jsonify(result)
  

@app.route('/lung',methods=['POST'])
def lung():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_lung.predict(arr)[0]
    accuracy=96.7
    result= {'result': int(num),'acc':accuracy}
    return jsonify(result)  
    

@app.route('/cerv_canc',methods=['POST'])
def cerv_canc():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_cerv.predict(arr)[0]
    accuracy=98.8
    result= {'result': int(num),'acc':accuracy}
    return jsonify(result) 


@app.route('/thyroid',methods=['POST'])
def thyroid():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_thyroid.predict(arr)[0]
    accuracy_thyroid=99.7
    result= {'result': int(num),'acc':accuracy_thyroid}
    return jsonify(result)
    
    
    
@app.route('/parkinsons',methods=['POST'])
def parkinsons():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_parkinsons.predict(arr)[0]
    accuracy_parkinsons=92.3
    result= {'result': int(num),'acc':accuracy_parkinsons}
    return jsonify(result)
    
    
@app.route('/age_chr_kid',methods=['POST'])
def age_chr_kid():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_chr_kid.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
    
  
@app.route('/age_cvd',methods=['POST'])
def age_cvd():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_cvd.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
    
    
@app.route('/age_diab',methods=['POST'])
def age_diab():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_diab.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
  
  
@app.route('/age_stroke',methods=['POST'])
def age_stroke():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_stroke.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
    
    
@app.route('/age_thyroid',methods=['POST'])
def age_thyroid():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_thyroid.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
    

@app.route('/age_lung',methods=['POST'])
def age_lung():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_lung.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)
        


@app.route('/age_cerv_canc',methods=['POST'])
def age_cerv_canc():
    data = request.json['data']
    arr = np.array(data).reshape(1, -1)
    num = model_age_cerv.predict(arr)[0]
    result = {'result': int(num)}
    return jsonify(result)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)