import time
from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def printInput():
    input = request.form['input']
    print(input, flush=True)
    return "test"
