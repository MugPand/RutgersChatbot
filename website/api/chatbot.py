import numpy as np
import pandas as pd
import re
import sys
import os
import spacy
import random
import math
import warnings
import sklearn.svm
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.neighbors
import sklearn.ensemble
import sklearn.preprocessing

trueoriginal = sys.stdout
outfile = '../../database/out.txt'
databasefile = '../../database/CSClass_info.csv'
courseregex = r"\d{2}:\d{3}:\d{3}"
coursematcher = re.compile(courseregex)


# outfile = '/content/drive/MyDrive/IRTStuff/out.txt'
#databasefile = '/content/drive/MyDrive/IRT/CSClass_info.csv'



# if os.path.exists(str(outfile)):
#     os.remove(outfile)

open(outfile, 'w').close()

# %% [markdown]
# Prereq stuff

# %%
def calc_prereqs(taken, check_course):
    data=pd.read_csv(databasefile)
    row = data['Course Number']
    # print(row)
    tempPrereq = data.loc[data['Course Number']==check_course]['Prereq Info']     #from database
    # print(tempPrereq)
    # print(pd.isna(tempPrereq))
    if tempPrereq.size == 0 or pd.isna(tempPrereq.iloc[0]):
      tempPrereq = ''
    else:
      tempPrereq = tempPrereq.iloc[0].replace('.','')
    # prereq = tempPrereq.split(";")
    # print(tempPrereq)
    prereq = re.split(r';\xa0|; ', tempPrereq)

    tempPrev = "; ".join(taken)       #from input
    stud_prev = tempPrev.split("; ")

    # print('prereq', prereq)
    # print('stud_prev', stud_prev)

    ret = []

    for i in range (len(prereq)):
        temp = prereq[i]

        if len(temp) == 10:
            contains = False
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    contains = True
            if contains == False:
                ret.append(temp)

        # else:
        #     temp_list = re.split(r"\xa0or\xa0", temp)
        #     temp_ret = []

        #     for j in range (len(temp_list)):
        #         temp_item = temp_list[j]
        #         contains = False
        #         for k in range (len(stud_prev)):
        #             if stud_prev[k] == temp_item:
        #                 contains = True
        #         if contains == False:
        #             temp_ret.append(temp_item)

        #     if len(temp_ret) != 0:
        #         ret.append(temp_ret)

        else:
            temp_list = re.split(r"\xa0or\xa0", temp)
            temp_ret = []

            contains = False
            for j in range (len(temp_list)):
                temp_item = temp_list[j]
                # contains = False
                for k in range (len(stud_prev)):
                    if stud_prev[k] == temp_item:
                        contains = True
            if contains == False:
                temp_ret.append(temp_item)

            if len(temp_ret) != 0:
                ret.append(temp_ret)

    # print()
    return ret

# %%
database = pd.read_csv(databasefile)
# print(database)
def retrieve(course, identifier):
    i = database[(database['Course Number']==course)]
    if len(i) == 0:
        return None
    i = i[identifier].item()
    return i

# %%
# print(database)
# print(retrieve('01:198:419', 'Credits'))

# %%
# Helper methods
class State:
    currQ = None
    running = True
    inputhistory = []
    outputhistory = []
    qhistory = []
    courselist = None
    def __init__ (self, q):
        self.setQuery(q)
    def setQuery(self, q):
        self.currQ = q
        self.qhistory.append(q)

    #In progress
    def descend(self, q):
        return
    def ascend(self, q):
        return

def redirect_to_file(text):
    original = sys.stdout
    x = open(outfile, 'a')
    sys.stdout = x
    if original == sys.stdout:
        original = trueoriginal
        # print('this is your redirected text:')
    print(text)
    sys.stdout = original
    x.close()
    # with open(outfile, 'a') as f:
    #     with redirect_stdout(f):
    #         print(text)


def fakeprint(state, *args):
    a = list(map(str, args))
    s = ' '.join(a)
    redirect_to_file(s)
    state.outputhistory.append(s)
    # print(s)
    return s


def farewellmethod(state):
    fakeprint(state, 'goodbye then!')
    state.running = False

def courseaccess(state):
    state.courselist = state.courselist if state.courselist else []
    courselist = state.courselist
    x = input()
    while x != 'done':
        courselist.append(x)
        x = input()
    fakeprint(state, "Ok, thanks! I'll remember those courses")

def cantake(state):
    print("test")
    if not state.courselist:
      fakeprint(state, "I don't know what courses you have, I'm afraid. What are they?")
      state.setQuery(queries['courses'])
      courseaccess(state)
      state.setQuery(queries['cantake'])
    fakeprint(state, 'your current courses are', state.courselist)
    prereqs = calc_prereqs(state.courselist, coursematcher.findall(state.inputhistory[-1])[-1])
    # fakeprint(state, "So, I have no idea whether you can take this one!")
    fakeprint(state, 'You can take this course!' if len(prereqs) == 0 else 'You still need '+str(prereqs))
    return
# %%
nlp = spacy.load("en_core_web_lg")


# %%
class Query:
    name = ""
    examples = []

    @staticmethod
    def response():
       print(tempvar)

    children = {}
    model = None
    decoder = None
    responsegenerator = {}

    def responses(self):
      return {'response': getattr(self,'response')}

    def __init__(self, name, examples, response, children = []):
        self.name = name
        self.examples = [nlp(x) for x in examples]
        self.response = response
        self.responsegenerator['response'] = classmethod(response)
        test = [[(e.text, e.label_) for e in d.ents] for d in self.examples]
        print(test)
        self.children = dict(zip(map(lambda x : x.name, children), children))
        if len(self.children) > 0:
            self.sktrainQuery()

    def addChild(self, child, suppressTrain=False):
        if child.name in self.children:
            return None

        self.children[child.name] = child

        if not suppressTrain:
            self.sktrainQuery()

    def addChildren(self, childlist):
        for childQ in childlist:
            self.addChild(childQ, suppressTrain=True)

        self.sktrainQuery()

    def removeChild(self, name, suppressTrain=False):
        self.children.discard(name)

        if not suppressTrain:
            self.sktrainQuery()

    def removeChildren(self, namelist):
        for name in namelist:
            self.removeChild(name, suppressTrain=True)

        if len(namelist) > 0:
            self.sktrainQuery()



    def trainingData(self):
        # return [[s, self.name] for s in [x.vector.reshape(1,-1) for x in self.examples]]
        return [[s, self.name] for s in self.examples]

    def queriessqrt(self):
        return round(math.sqrt(len(self.children)))

    def getModel(self):
        print('getting model')
        rfc = sklearn.ensemble.RandomForestClassifier(max_features='auto')
        # gnbc = sklearn.naive_bayes.GaussianNB()
        sgdc = sklearn.linear_model.SGDClassifier()
        ptronc = sklearn.linear_model.SGDClassifier(loss='perceptron')
        # knc = sklearn.neighbors.KNeighborsClassifier()


        # combinedclassifier = sklearn.naive_bayes.GaussianNB()
        combinedclassifier = sklearn.linear_model.SGDClassifier(penalty='elasticnet', loss='hinge')

        ensemblec = sklearn.ensemble.StackingClassifier(estimators=[('rf', rfc),
                                                                    ('sgd', sgdc),
                                                                    ('ptron', ptronc),
                                                                    # ('gnb', gnbc),
                                                                    # ('kn', knc)
                                                                    ],
                                                        final_estimator=combinedclassifier)

        params = {'cv': [None, 2],
                  'rf__n_estimators': [100, self.queriessqrt()],
                  # 'rf__criterion': ['gini', 'entropy'],
                  # 'rf__max_features': ['auto', 'log2'],
                  # 'rf__warm_start':['true'],
                  'rf__class_weight':[None, 'balanced'],
                  # 'sgd__loss': ['hinge', 'log'],
                  'sgd__penalty': ['l2', 'elasticnet'],
                  # 'kn__n_neighbors': [5, self.queriessqrt()],
                  'final_estimator__warm_start': [False]
                  }
        gc = sklearn.model_selection.GridSearchCV(ensemblec, params)

        return sgdc


    def sktrain(self, datalist, le=None, vc=None):
        # print(datalist)
        train = datalist
        target = train[train.columns[len(train.columns)-1]]
        train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
        if le is None:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(target)
        target = le.transform(target)
        # print(train.values.tolist())
        # print(train)
        # train = pd.DataFrame(list(train.apply(lambda x : x.vector.reshape(1,-1), axis=1, result_type='reduce')))
        train = np.stack([d[0][0] for d in train.values.tolist()])
        # print("train")
        # print(train)
        # print("target")
        # print(target)
        # print()
        if vc is None:
            vc = sklearn.linear_model.SGDClassifier()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vc = self.getModel()
        vc.fit(train, target)


        # vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(input='content', use_idf=True, ngram_range=(1,2))

        self.model = vc
        self.decoder = le
        return vc, le



    def sktrainQuery(self):
        queries = self.children.values()
        classifier = None
        le = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = pd.concat([] if len(queries)==0 else [pd.DataFrame(q.trainingData(), columns=['data', 'target']) for q in queries])
            data.reset_index(drop=True, inplace=True)
            data['data'] = pd.Series([x.vector.reshape(1,-1) for x in data['data'].tolist()])
            # print(data)
            classifier, le = self.sktrain(data)
            # print('error not there')
        return classifier, le

    def sktestQuery(self, cl=None, le=None):
        queries = self.children.values()
        data = pd.concat([] if len(queries)==0 else [pd.DataFrame(q.trainingData(), columns=['data', 'target']) for q in queries])
        # datalist = pd.DataFrame([q.trainingData() for q in list(queries.values())], columns = ['data', 'target'])
        data.reset_index(drop=True, inplace=True)
        datalist = data.copy()
        datalist['data'] = pd.Series([x.vector.reshape(1,-1) for x in data['data'].tolist()])

        traindatalist = pd.DataFrame(columns=['data', 'target'])
        testdatalist = pd.DataFrame(columns=['data', 'target'])

        idxs = random.sample(range(len(datalist)), min(round(0.6*len(datalist)), len(datalist)))
        traindatalist = datalist.iloc[idxs]
        testdatalist = datalist.drop(labels=idxs, axis=0)

        print('training...')
        cl, le = self.sktrain(traindatalist, vc=cl, le=le)

        testdatalist['target'] = le.transform(testdatalist['target'])


        score = cl.score(np.stack([d[0] for d in testdatalist['data'].values.tolist()]), testdatalist['target'])

        print("sklearn classifier scored " + str(score))
        # print("data was")
        # print(data)
        # print()
        # print('test indices are ')
        # print(testdatalist.index)
        # print(data['data'][1])
        data.columns = ['dataname', 'target']

        printtestdatalist = testdatalist.copy()
        printtestdatalist['target'] = le.inverse_transform(testdatalist['target'])
        printtestdatalist = printtestdatalist.join(data.drop('target', axis=1))
        print(printtestdatalist)
        print()

        return score, traindatalist, testdatalist, datalist



# %%
queries = {'root': Query('root',
                         ['home',
                          'cancel',
                          'cancel query',
                          'clear',
                          'I have another question',
                          'I want to know something else',
                          'Ok, thanks!',
                          'Thanks',
                          'thanks'
                          'next question'],
                         lambda state: fakeprint(state, 'What would you like to know?')),
          'greeting': Query('greeting',
                             ["hello, how are you?",
                              "greetings!",
                              "salutations!",
                              "nice to meet you!",
                              "good morning!",
                              "hello.",
                              "hi.",
                              "hi!"],
                             lambda state: (fakeprint(state, "hello to you too!"),
                                            # None if 'greeting' not in childrenmap['root'] else childrenmap['root'].remove('greeting'),
                                            state.setQuery(queries['root'] if len(state.qhistory)<=1 else state.qhistory[-2]))),
           'farewell': Query('farewell',
                             ["goodbye.",
                              "goodbye!",
                              "see you later!",
                              "see you later, then!",
                              "farewell!",
                              "bye then!",
                              "ok, bye then!"],
                             lambda state: farewellmethod(state)),
           'whatclass': Query('whatclass',
                              ['What course should I take?',
                               'What course does it make sense for me to take?',
                               'What course should I add?',
                               'What course deals with Z?',
                               'What course can teach me about Z?',
                               'What class do you suggest?'],
                              lambda state: fakeprint(state, "Why are you asking a computer about your course choices? Ask an advisor")),
           'shouldtake': Query('shouldtake',
                               ['Does it make sense for me to take Discrete?',
                                'Is Discrete applicable for the Y minor?',
                                'Should I add Discrete?',
                                'Should I take Discrete?',
                                'Do I have to take Discrete?',
                                'Do I need to take Discrete?',
                                'Is Discrete required for the Y minor?',
                                'Would you recommend Discrete?',
                                'What do you think about Discrete?',
                                'What about Discrete?',
                                'How is Discrete?'],
                               lambda state: fakeprint(state, "Why are you asking a computer about your course choices? Ask an advisor!")),
           'shoulddrop': Query('shoulddrop',
                               ['Should I drop Discrete?',
                                'Should I drop Discrete course?',
                                'I don\'t think I should take Discrete',
                                'Can I avoid taking Discrete?',
                                'Am I able to drop Discrete?',
                                'Will I have to take Discrete or can I drop it?',
                                'Will I have to take Discrete or can I skip it?',
                                'What course can I switch for Discrete?',
                                'I don\'t think I should take Discrete. What could I take instead?',
                                'And without Discrete?',
                                'What course can I swap for Discrete?'],
                               lambda state: fakeprint(state, "I have no idea, I'm not a person! Advisors are, though!")),
           'cantake': Query('cantake',
                            ['Can I take 01:198:205?',
                             'Can I take 01:198:206?',
                             'Can I register for 01:198:206?',
                             'Is it possible to take 01:198:205?'],
                            lambda state: cantake(state)),
           'info': Query('info',
                         ['Tell me about 01:198:105',
                          'Can you tell me about 01:198:101?',
                          'Can you tell me about 01:198:101',
                          'And about 01:198:203?'],
                         lambda state: (fakeprint(state, *(coursematcher.findall(state.inputhistory[-1])[-1],
                                                           '\n name: ' + str(retrieve(coursematcher.findall(state.inputhistory[-1])[-1], 'Name')),
                                                           '\n credits: ' + str(retrieve(coursematcher.findall(state.inputhistory[-1])[-1], 'Credits')),
                                                           '\n professor: ' + str(retrieve(coursematcher.findall(state.inputhistory[-1])[-1], 'Professor'))
                                                          # 'name: ' + str(retrieve(state.inputhistory[-1].split(' ')[-1], 'Name')),
                                                          # 'credits: ' + str(retrieve(state.inputhistory[-1].split(' ')[-1], 'Credits')),
                                                          # 'professor:'+ str(retrieve(state.inputhistory[-1].split(' ')[-1], 'Professor'))
                                                          ))
                         if len(coursematcher.findall(state.inputhistory[-1])) > 0
                         and [retrieve(coursematcher.findall(state.inputhistory[-1])[-1], x) for x in ['Name', 'Credits', 'Professor']] != [None, None, None]
                         else
                         fakeprint(state, "I'm afraid " + ("this course" if not coursematcher.search(state.inputhistory[-1]) else coursematcher.findall(state.inputhistory[-1])[-1]) + " doesn't exist"),
                         None
                          # if len(state.qhistory)<=1 else state.setQuery(state.qhistory[-2])
                         )),
           'courses': Query('courses',
                            ['Here are my courses',
                             "I'll tell you what courses I've taken",
                             "I've previously taken these courses"],
                            lambda state: courseaccess(state)),
           'lastq': Query('lastq',
                          ['And 01:198:111?',
                           'And 01:198:314?',
                           'And 01:198:206?',
                           'And 01:198:205?',
                           'Can you do that for 01:198:302?',
                           'Can you do that for 01:198:112?',
                           'Can you do that for 01:198:305?',
                           'Can you do that for 01:198:205?',
                           '01:198:512, then?',
                           '01:198:111, then?',
                           '01:198:211, then?',
                           '01:198:109, then?'],
                          lambda state: (fakeprint("There's nothing for me to repeat...")
                          if len(list(filter(lambda x: x.name != 'root', state.qhistory)))<=1
                          else state.setQuery(list(filter(lambda x : x.name != 'root', reversed(state.qhistory)))[0 if state.currQ.name=='root' else 1]),
                                        #  print('lastq on ' + state.inputhistory[-1] + 'with query ' + state.currQ.name),
                                         state.currQ.responses()['response'](state) if len(state.qhistory)<=1 else None))
           }


childrenmap = {'root': ['root', 'greeting', 'greeting', 'farewell', 'shouldtake', 'shoulddrop', 'cantake', 'whatclass', 'info', 'courses', 'lastq'],
               'greeting': ['root', 'farewell', 'shouldtake', 'shoulddrop', 'cantake', 'whatclass', 'info', 'courses'],
               'shoulddrop': ['farewell', 'shouldtake', 'shoulddrop', 'cantake', 'whatclass', 'lastq'],
               'shouldtake': ['farewell', 'shouldtake', 'shoulddrop', 'cantake', 'whatclass', 'lastq'],
               'cantake': ['farewell', 'shouldtake', 'cantake', 'whatclass', 'lastq'],
               'whatclass': ['farewell', 'shouldtake'],
               'info': ['info', 'lastq', 'root', 'farewell', 'lastq'],
               'courses': ['greeting', 'farewell']}

for k in childrenmap:
    queries[k].addChildren(queries[a] for a in childrenmap[k])
# for q in queries:
#     queries[q].sktrainQuery()

# queries['cantake'].responses = lambda self : print(self.name)
# queries['cantake'].responses()

# %%

def chatbot(state):
    statement = nlp(state.inputhistory[-1])
    # print(statement)

    # currQ = queries['greeting']

    likelihoods = dict(map(lambda a: (a, max([statement.similarity(x) for x in state.currQ.children[a].examples])), state.currQ.children))
    # print(likelihoods)
    try:
        largest = max(likelihoods, key=likelihoods.get)
    except:
        largest = None
    # print(largest)
    # currQ.children.get(largest, queries['farewell']).responses()['response']()

    cl = state.currQ.model
    le = state.currQ.decoder

    # print("sklearn")
    # print(le.inverse_transform(cl.predict([statement.vector])))
    # print("sklearn probabilities")
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         prob = cl.predict_proba([statement.vector]).tolist()[0]
    #         print(prob)
    #         print(le.inverse_transform(range(len(prob))))
    #         print(dict(zip(le.inverse_transform(range(len(prob))), prob)))
    # except:
    #     print("probabilities not available")

    state.setQuery(queries.get(le.inverse_transform(cl.predict([statement.vector]))[0], 'farewell'))
    outputstring = state.currQ.responses()['response'](state)
    # outputhistory.append(outputstring)

tempvar = None # current input

def run():
    s = State(queries['root'])
    s.running = True
    # leaverlist = queries['farewell'].examples
    # leaverlist = leaverlist + [nlp(x) for x in ["quit.", "quit", "exit", "leave", "goodbye"]]
    while s.running:
        print(s.currQ.name)
        print(s.currQ.children.keys())
        s.inputhistory.append(input("say something!   "))
        # inputhistory.append(tempvar)
        print()
        chatbot(s)
        print()
        # print("leaver max:")
        # print(max([nlp(tempvar).similarity(leaver) for leaver in leaverlist]))
        # print()
        # print()
    return s

# %%

# currstate = run()
# print(currstate.inputhistory)
# print(currstate.outputhistory)
# print(currstate.courselist)

# # %%
# tempvar = "Tell me about Data Structures"
# temp = "Data Structures"
# # retrieve(temp, 'credits')
# # (lambda : print(retrieve(tempvar[14:], 'credits')))()
# x = queries['info'].responses()['response']()

# # %%
# df = pd.read_csv('smallData.csv')
# df.head(5)

# # %% [markdown]
# # Define retrieve function for accessing information in database

# # %%
# def retrieve(course, identifier):
#     i = df[(df['id']==course) | (df['name']==course)]
#     i = i[identifier].item()
#     return i

# # %%
# # retrieve("01:198:112", "credits")

# # %% [markdown]
# # Load Spacy Model & Tokenize utterances

# # %%
# nlp = spacy.load("en_core_web_sm")
# def tokenize(utterance):
#     doc = nlp(utterance)
#     tagged_tokens = []
#     for token in doc:
#         tagged_tokens.append((token.text, token.pos_))

#     return tagged_tokens

# # %%
# tokenize("I am very happy!")

# # %% [markdown]
# # Handle Tagged & Untagged Reflections

# # %%
# tagged_reflection_of = {
#     ("you", "PPSS") : "I",
#     ("you", "PPO") : "me"
# }

# untagged_reflection_of = {
#     "am"    : "are",
#     "i"     : "you",
#     "i'd"   : "you would",
#     "i've"  : "you have",
#     "i'll"  : "you will",
#     "i'm"   : "you are",
#     "my"    : "your",
#     "me"    : "you",
#     "you've": "I have",
#     "you'll": "I will",
#     "you're": "I am",
#     "your"  : "my",
#     "yours" : "mine"
# }

# # %% [markdown]
# # Translate Tokens

# # %%
# def translate_token(wt) :
#     (word, tag) = wt
#     wl = word.lower()
#     if (wl, tag) in tagged_reflection_of :
#         return (tagged_reflection_of[wl, tag], tag)
#     if wl in untagged_reflection_of :
#         return (untagged_reflection_of[wl], tag)
#     if tag.find("NP") < 0 :
#         return (wl, tag)
#     return (word, tag)

# subject_tags = ["PPS",  # he, she, it | pronouns singular
#                 "PPSS", # you, we, they | pronouns plural
#                 "PN",   # everyone, someone | indefinite pronoun
#                 "NN",   # dog, cat | noun, common, singular or mass
#                 "NNS",  # dogs, cats | noun, common, plural
#                 "NP",   # Fred, Jane | proper noun
#                 "NPS"   # Republicans, Democrats | proper plural
#                 ]

# def swap_ambiguous_verb(tagged_words, tagged_verb_form, target_subject_pronoun, replacement) :
#     for i, (w, t) in enumerate(tagged_words) :
#         if (w, t) == tagged_verb_form :
#             j = i - 1
#             # look earlier for the subject
#             while j >= 0 and tagged_words[j][1] not in subject_tags :
#                 j = j - 1
#             # if subject is the target, swap verb forms
#             if j >= 0 and tagged_words[j][0].lower() == target_subject_pronoun :
#                 tagged_words[i] = replacement
#             # didn't find a subject before the verb, so probably a question
#             if j < 0 :
#                 j = i + 1
#                 while j < len(tagged_words) and tagged_words[j][1] not in subject_tags :
#                     j = j + 1
#                 # if subject is the target, swap verb forms
#                 if j < len(tagged_words) and tagged_words[j][0].lower() == target_subject_pronoun :
#                     tagged_words[i] = replacement

# def handle_specials(tagged_words) :
#     # don't keep punctuation at the end
#     while tagged_words[-1][1] == 'PUNCT' :
#         tagged_words.pop()
#     # replace verb "be" to agree with swapped subjects
#     swap_ambiguous_verb(tagged_words, ("are", "BER"), "i", ("am", "BEM"))
#     swap_ambiguous_verb(tagged_words, ("am", "BEM"), "you", ("are", "BER"))
#     swap_ambiguous_verb(tagged_words, ("were", "BED"), "i", ("was", "BEDZ"))
#     swap_ambiguous_verb(tagged_words, ("was", "BEDZ"), "you", ("were", "BED"))


# close_punc = ['.', ',', "''"]
# def translate(this):
#     '''tokens = tokenize(this)
#     tagged_tokens = tagger.tag(tokens)'''
#     tagged_tokens = tokenize(this)
#     print(tagged_tokens)
#     translation = [translate_token(tt) for tt in tagged_tokens]
#     handle_specials(translation)
#     if len(translation) > 0 :
#         with_spaces = [translation[0][0]]
#         for i in range(1, len(translation)) :
#             if translation[i-1][1] != '``' and translation[i][1] not in close_punc :
#                 with_spaces.append(' ')
#             with_spaces.append(translation[i][0])
#     return ''.join(with_spaces)

# # %%
# translate("I am very happy!")

# # %% [markdown]
# # Eliza based responses

# # %%
# rules = [(re.compile(x[0]), x[1]) for x in [
#     ['How are you?',
#          [ "I'm fine, thank you."]],
#     ['Thank you!',
#          ['No problem, did you have more questions?']],
#     ["I had a question about (.*)",
#          [  "Ask away!",
#             "I am happy to help!",
#             "Ok, What is your question?"]],
#     ["Hello(.*)",
#          [  "Hello... I'm glad you could drop by today.",
#             "Hi there... how are you today?"]],
#     ["I'm great",
#          ["Great! Did you want to ask me anything?",
#             "Wonderful! Feel free to ask me any questions that you may have!"]],
#     ["Can you answer (.*)",
#          [  "Yes, I can!",
#             "What is your question?"]],
#     ["quit",
#          [  "Thank you for talking with me.",
#             "Good-bye.",
#             "Have a good day!"]],
#     ["(.*)",
#          [  "Please tell me more.",
#         "Can you elaborate on that?", "I don't understand. Can you rephrase? "]]
# ]]

# def respond(sentence):
#     # find a match among keys, last one is quaranteed to match.
#     for rule, value in rules:
#         match = rule.search(sentence)
#         if match is not None:
#             # found a match ... stuff with corresponding value
#             # chosen randomly from among the available options
#             resp = random.choice(value)
#             # we've got a response... stuff in reflected text where indicated
#             while '%' in resp:
#                 pos = resp.find('%')
#                 num = int(resp[pos+1:pos+2])
#                 resp = resp.replace(resp[pos:pos+2], translate(match.group(num)))
#             return resp

# # %%
# respond("I had a question about computer science?")

# # %%
# respond("Hello, I wanted to ask about computer science.")

# # %%
# respond("Thank you!")

# # %% [markdown]
# # Build course suggestion tree

# # %%
# # define basic node class for course suggestion tree
# class Node(object):
#     def __init__(self, val):
#         self.val = val
#         self.children = []

#     def add_child(self, obj):
#         self.children.append(obj)

# q1 = Node("suggesting courses, overview of CS or narrowed focus?")
# q2 = Node("You picked overview of CS! academic discipline or applications?")
# q3 = Node("You picked narrowed focus! software engineering or data science?")
# q4 = Node("You picked applications! math & physical sciences or digital creation?")
# q5 = Node("You picked software engineering! beginner or experienced?")
# c1 = Node("You picked academic discipline! 01:198:105	Great Insights in Computer Science")
# c2 = Node("You picked math & physical sciences! 01:198:107	Computing for Math and the Sciences")
# c3 = Node("You picked digital creation! 01:198:110	Principles of Computer Science")
# c4 = Node("You picked data science! 01:198:142	Data 101: Data Literacy")
# c5 = Node("You picked beginner! 01:198:111	Introduction to Computer Science")
# c6 = Node("You picked experienced! 01:198:112	Data Structures")
# q1.add_child(q2)
# q1.add_child(q3)
# q2.add_child(c1)
# q2.add_child(q4)
# q4.add_child(c2)
# q4.add_child(c3)
# q3.add_child(c4)
# q3.add_child(q5)
# q5.add_child(c5)
# q5.add_child(c6)

# # %% [markdown]
# # Find possible actions for robot

# # %%
# def possible(state):
#     # checking for eliza rules or task state
#     plans = []

#     # check if 0 utterances have been said
#     if(len(state[utterances]) == 0):
#         plans.append("Type to chat!")
#     else:
#         # gets eliza response
#         plans.append(respond(state[utterances][most_recent]))


#         # check if no tasks have been initialized
#         # q1Rules = ["suggest", "class", "course", "recommend"]
#         q1Rules = ["recommend"]
#         if(len(state[tasks]) == 0):
#             for word in state[utterances][most_recent].split():
#                 if word in q1Rules:
#                     # update_state(q1, state)
#                     plans.append(q1)
#                     break
#         # check if tasks have been initialized -> add the next appropriate task
#         elif(isinstance(state[tasks][most_recent], Node)):
#             for child in state[tasks][most_recent].children:
#                 if state[tasks][most_recent] in child.val:
#                     # update_state(child, state)
#                     plans.append(child)
#                     break


#         # information retrieval "tell me information about X"
#         # infoRetrievalRules = ["information", "tell me", "describe"]
#         infoRetrievalRules = ["information"]
#         for rule in infoRetrievalRules:
#             if rule in state[utterances][most_recent].lower():
#                 course = None
#                 identifier = None
#                 for id in df['id'].tolist():
#                     if id.lower() in state[utterances][most_recent].lower():
#                         course = id
#                         break
#                 if course == None:
#                     for name in df['name'].tolist():
#                         if name.lower() in state[utterances][most_recent].lower():
#                             course = name
#                             break
#                 for i in df.columns:
#                     if i.lower() in state[utterances][most_recent].lower():
#                         identifier = i
#                         break
#                 plans.append(retrieve(course, identifier))


#     return plans

# # %% [markdown]
# # Find possible actions for user

# # %%
# def userUtterance(state):
#     # default plans
#     plans = ["quit", "Mistake"]
#     # respond to suggestion
#     if len(state[tasks]) > 0 and len(state[tasks][most_recent].children) != 0:
#         tokens = re.split(", |! | or ", state[tasks][most_recent].val)
#         plans = plans + tokens[-2:]
#     # ask for suggestion
#     if len(state[tasks]) == 0:
#         # q1Rules = ["suggest", "class", "course", "recommend"]
#         q1Rules = ["recommend"]
#         plans.append(random.choice(q1Rules))

#     #infoRetrievalRules = ["information", "tell me", "describe"]
#     infoRetrievalRules = ["information"]
#     #courses = []
#     #courses.append(df["id"])

#     # ask about class
#     courses = list(df["id"]) + list(df["name"])
#     #print(courses)
#     utterance = random.choice(infoRetrievalRules) + " " + random.choice(df.columns) + " " + random.choice(courses)
#     plans.append(utterance)

#     return plans


# # %% [markdown]
# # Update States

# # %%
# # initial_state = s0 = {"u": [], "a": [], "task": []}
# initial_state = s0 = ((),(),())

# # %%
# # define constants
# utterances = 0
# actions = 1
# tasks = 2
# most_recent = -1

# # %%
# def update_state(task, state):
#     #     state.get("task").append(task)
#     #     return state

#     u, a, t = list(state)
#     t = list(t)
#     t.append(task)
#     t = tuple(t)
#     newState = (u, a, t)
#     return newState


# # s1 = update_state("hello", s0)
# # print(s1)

# # %%
# def do(item, state):
# #     n = state.copy()
# #     n['a'] = n['a'].copy()
# #     n['a'].append(item)

#     u, a, t = list(state)
#     a = list(a)
#     a.append(item)
#     a = tuple(a)
#     newState = (u, a, t)
#     return newState

# # s1 = do("hello", s0)
# # print(s1)

# # %%
# def understand(item, state):
# #     # cases where state needs to be updated
# #     n = state.copy()
# #     # check if string is not empty
# #     if item:
# #         n['u'] = n['u'].copy()
# #         n['u'].append(item)

#     u, a, t = list(state)
#     u = list(u)
#     if item:
#         u.append(item)
#     u = tuple(u)
#     newState = (u, a, t)
#     return newState

# # s1 = understand("hello", s0)
# # print(s1)

# # %% [markdown]
# # Decide random action

# # %%
# def deliberate(plans, state):
#     choice = random.choice(plans)
#     if isinstance(choice, Node):
#         state = update_state(choice, state)
#         return choice.val
#     else:
#         return choice

# # %% [markdown]
# # Converse with list of utterances

# # %%
# def converse(utterances):
# #     s0 = {"u": [], "a": [], "task": []}
#     s0 = ((), (), ()) # represents utterances, actions, tasks
#     for utterance in utterances:
#         if utterance == "quit":
#             break
#         s1 = understand(utterance, s0)
#         plans = possible(s1)
#         action = deliberate(plans, s1)
#         s0 = do(action, s1)
#         print(utterance)
#         print(action)

# # %%
# converse(["Hello", "I had a question about computer science", "can you suggest a course that I would like?", "overview", "academic", "quit"])

# # %% [markdown]
# # Converse with user

# # %%
# utterance = None
# # s0 = {"u": [], "a": [], "task": []}
# s0 = ((), (), ()) # represents utterances, actions, tasks
# while(utterance != "quit"):
#     s1 = understand(utterance, s0)
#     #print(s1)
#     plans = possible(s1)
#     action = deliberate(plans, s1)
#     s0 = do(action, s1)
#     #print(s0)
#     utterance = input(action)

# # %% [markdown]
# # Simulate conversation between user and chatbot

# # %%
# def simulate(state0):
#     utterance = None
#     while(utterance != "quit"):
#         state1 = understand(utterance, state0)
#         #print(state1)
#         plans = possible(state1)
#         action = deliberate(plans, state1)
#         state0 = do(action, state1)
#         #print(state0)
#         print("Action: " + str(action))
#         utterance = random.choice(userUtterance(state0))
#         print("Uterrance: " + utterance)

# # %%
# # s0 = {"u": [], "a": [], "task": []}
# s0 = ((), (), ()) # represents utterances, actions, tasks
# simulate(s0)

# # %% [markdown]
# # # Machine Learning

# # %% [markdown]
# # Define reward function

# # %%
# leafNodes = [c1, c2, c3, c4, c5, c6]
# def getReward(oldState, action, newState):
#     reward = -1
#     # action is in dataframe
#     if (action in df.values):
#         reward += 5
#     else:
#         # action is a leaf node
#         for node in leafNodes:
#             if str(action) in node.val:
#                 reward += 5

#     return reward

# # %%
# # getReward(4), getReward("Data Structures")

# # %% [markdown]
# # Determine if state is terminated

# # %%
# def is_terminated(state):
#     if len(state[utterances]) == 0:
#         return False
#     elif state[utterances][most_recent] == "quit":
#         return True
#     return False

# # %% [markdown]
# # Define Q-actions

# # %%
# def q_action(state, actions, epsilon):
#     if random.random() < epsilon:
#         max_q = None
#         for action in actions:
#             #check to see if state action pair is in q_values dictionary, if not insert it
#             if (state, action) not in q_values:
#                 q_values.setdefault((state, action), 0)

#             if q_values.get(state,action) > q_values.get(max_q):
#                 max_q = (state,action)

#         return max_q[1]

#     else:
#         return random.choice(actions)

# # %% [markdown]
# # Training

# # %%
# # dicitionary to hold Q-values for each state & action pair(key)
# q_values = {}
# # {(state, action) : q_value}

# # define training params
# epsilon = 0.9
# discount_factor = 0.9
# learning_rate = 0.9

# default = 4

# for episode in range(100):
#     # begin conversation and iterate till it finishes
#     # state0 = {"u": [], "a": [], "task": []}
#     state0 = ((), (), ())
#     utterance = None

#     while(True):
#         # determine chatbot action
#         plans = possible(state0)
#         action = deliberate(plans, state0)
#         userTurnState = do(action, state0)

#         # determine user utterance
#         utterance = random.choice(userUtterance(userTurnState))
#         # print(utterance)
#         nextState = understand(utterance, userTurnState)

#         # handle rewards (reward for action that transitions to nextState)
#         reward = getReward(state0, action, nextState)
#         old_q_value = q_values.get((state0, action), default)


#         # if newState is not in dictionary, add it
#         found = False
#         for k in q_values:
#             if(k[0] == nextState):
#                 found = True
#                 break
#         if not found:
#             q_values[(nextState, None)] = default


#         # get q_values associated with next state
#         vals = []
#         for key, val in q_values.items():
#             if(key == nextState):
#                 vals.append(val)
#         if(len(vals) == 0): vals.append(default)
#         td = reward + (discount_factor * max(vals) - old_q_value)
#         newqq = old_q_value + learning_rate * (td)

#         q_values.update({(state0, action):newqq})

#         if utterance == "quit":
#             break

#         state0 = nextState
#         # print(state0)

# print("Training complete!")
# # print q-value dictionary
# for key, value in sorted(q_values.items(), key=lambda kv: kv[1], reverse=True):
#         print(key,' : ',value)
