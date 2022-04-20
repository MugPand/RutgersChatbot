from flask import Flask, request
from flask_cors import CORS, cross_origin
import chatbot as cb


app = Flask(__name__)
CORS(app, resources={r"/api/*":{"origins":"*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

s = cb.State(cb.queries['greeting'])


@app.route('/user', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def user():
    global s
    data = request.json['msg']
    s.inputhistory.append(data)
    cb.chatbot(s)
    if(s.outputhistory[-1] == 'goodbye then!'):
        temp = s.outputhistory[-1]
        s = cb.State(cb.queries['greeting'])
        return str(temp)

    return str(s.outputhistory[-1])
