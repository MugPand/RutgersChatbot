from flask import Flask, request
from flask_cors import CORS, cross_origin
import chatbot as cb


app = Flask(__name__)
CORS(app, resources={r"/api/*":{"origins":"*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

s = cb.State(cb.queries['greeting'])
courseRoutine = False
courseList = []
storeCourse = ''

@app.route('/user', methods=['GET','POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def user():
    global s, courseRoutine, courseList, storeCourse
    data = request.json['msg']
    s.inputhistory.append(data)
    output = ''
    if(courseRoutine):
        if data != 'done':
            courseList.append(data)
            output = 'Enter more courses or type done if all taken courses have been entered'
        else:
            courseRoutine = False
            cb.fakeprint(s, "Ok, thanks! I'll remember those courses")
            output += "Ok, thanks! I'll remember those courses\n"
            s.setQuery(cb.queries['cantake'])
            cb.fakeprint(s, 'your current courses are', s.courselist)
            output += ('your current courses are: ' + str(s.courselist) + '\n')
            prereqs = cb.calc_prereqs(s.courselist, storeCourse)
            cb.fakeprint(s, 'You can take this course!' if len(prereqs) == 0 else 'You still need '+str(prereqs))
            output += ('You can take this course!' if len(prereqs) == 0 else 'You still need '+str(prereqs))
    else:
        cb.chatbot(s)
        if(s.outputhistory[-1] == 'goodbye then!'):
            temp = s.outputhistory[-1]
            s = cb.State(cb.queries['greeting'])
            return str(temp)
        elif(s.outputhistory[-1] == 'I don\'t know what courses you have, I\'m afraid. What are they?'):
            courseRoutine = True
            storeCourse = cb.coursematcher.findall(s.inputhistory[-1])[-1]
            s.courselist = s.courselist if s.courselist else []
            courseList = s.courselist
            s.setQuery(cb.queries['courses'])
            return str(s.outputhistory[-1])

        output = str(s.outputhistory[-1])


    return output
