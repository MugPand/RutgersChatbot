import time
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/api/*":{"origins":"*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/user', methods=['POST'])
@cross_origin()
def user():
    data = request.json['msg']
    return str("test")
