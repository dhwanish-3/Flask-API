
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET','FETCH'])
def hello_world():
    return '[{"id":"18","date":"2023-07-05","name":"The Biz Whiz Quiz","venue":"Institute of Management in Kerala","imageUrl":"https:\/\/qcollective.in\/wp-content\/uploads\/2023\/07\/Screenshot_20230703-154925.png","category":"college","type":"business","masters":"","contact":"9072906628","link":null,"rules":null},{"id":"19","date":"2023-07-09","name":"Catch 22 2.0","venue":"North Malabar Chamber of Commerce,Caltex, Kannur","imageUrl":"https:\/\/qcollective.in\/wp-content\/uploads\/2023\/07\/Screenshot_20230703-154538-2.png","category":"school","type":"general","masters":"","contact":"8547605064","link":null,"rules":null}]'

