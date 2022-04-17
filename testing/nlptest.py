import math
import random
import warnings
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.cluster
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.feature_extraction
import sklearn.feature_extraction.text
import sklearn.model_selection
import sklearn.preprocessing
import spacy
from spacy.lang.en import English
import spacy.tokenizer as tokenizer

#####REQUIREMENTS#####
# spaCy
# en_core_web_lg (from spacy)
#      Note: Can be replaced with a different model, but for best performance, should be one with word vectors
# sklearn
# pandas
# numpy

# Accuracy threshold for the test data
threshold = 0.6


# nlp = English()
nlp = spacy.load("en_core_web_lg")
# tokenizer = tokenizer.Tokenizer(nlp.vocab)

class Query:
    name = ""
    examples = []
    response = lambda self : print()
    
    def __init__(self, name, examples, response):
        self.name = name
        self.examples = [nlp(x) for x in examples]
        self.response = response
        test = [[(e.text, e.label_) for e in d.ents] for d in self.examples] 
        print(test)

    def trainingData(self):
        # return [[s, self.name] for s in [x.vector.reshape(1,-1) for x in self.examples]]
        return [[s, self.name] for s in self.examples]

queries = {'greeting': Query('greeting',
                             ["hello, how are you?",
                              "greetings!",
                              "salutations!",
                              "nice to meet you!",
                              "good morning!",
                              "hello.",
                              "hi.",
                              "hi!"],
                             lambda : print("hello to you too!")),
           'farewell': Query('farewell',
                             ["goodbye.",
                              "goodbye!",
                              "see you later!",
                              "see you later, then!",
                              "farewell!"],
                             lambda : print("goodbye to you too!")),
           'whatclass': Query('whatclass',
                              ['What course should I take?',
                               'What course does it make sense for me to take?',
                               'What course should I add?',
                               'What course deals with Z?',
                               'What course can teach me about Z?',
                               'What class do you suggest?'],
                              lambda : print("whatclass")),
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
                               lambda : print("shouldtake")),
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
                               lambda : print("shoulddrop")),
           'cantake': Query('cantake',
                            ['Can I take Discrete?',
                             'Do I have the prerequisites to take Discrete?',
                             'Can I register for Discrete?'],
                            lambda : print("cantake"))}
queriessqrt = round(math.sqrt(len(queries)))

def getModel():
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
              'rf__n_estimators': [100, queriessqrt],
              # 'rf__criterion': ['gini', 'entropy'],
              # 'rf__max_features': ['auto', 'log2'],
              # 'rf__warm_start':['true'],
              'rf__class_weight':[None, 'balanced'],
              # 'sgd__loss': ['hinge', 'log'],
              'sgd__penalty': ['l2', 'elasticnet'],
              # 'kn__n_neighbors': [5, queriessqrt],
              'final_estimator__warm_start': [False]
              }
    gc = sklearn.model_selection.GridSearchCV(ensemblec, params)

    return sgdc


def sktrain(datalist, le=None, vc=None):
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
            vc = getModel()
    vc.fit(train, target)

    
    # vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(input='content', use_idf=True, ngram_range=(1,2))

    return vc, le

    

def sktrainQuery(queries):
    classifier = None
    le = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = pd.concat([pd.DataFrame(q.trainingData(), columns=['data', 'target']) for q in list(queries.values())])
        data.reset_index(drop=True, inplace=True)
        data['data'] = pd.Series([x.vector.reshape(1,-1) for x in data['data'].tolist()])
        classifier, le = sktrain(data)
    return classifier, le

def sktestQuery(queries, cl=None, le=None):
    data = pd.concat([pd.DataFrame(q.trainingData(), columns=['data', 'target']) for q in list(queries.values())])
    # datalist = pd.DataFrame([q.trainingData() for q in list(queries.values())], columns = ['data', 'target'])
    data.reset_index(drop=True, inplace=True)
    datalist = data.copy()
    datalist['data'] = pd.Series([x.vector.reshape(1,-1) for x in data['data'].tolist()])

    traindatalist = pd.DataFrame(columns=['data', 'target'])
    testdatalist = pd.DataFrame(columns=['data', 'target'])

    idxs = random.sample(range(len(datalist)), min(round(0.6*len(datalist)), len(datalist)))
    traindatalist = datalist.iloc[idxs]
    testdatalist = datalist.drop(labels=idxs, axis=0)

    # while not all(traindatalist['target'].tolist().count(elem) > 2 for elem in datalist['target'].tolist()):
    #     traindatalist = datalist.sample(frac=0.6, replace=False)

    # while not all(testdatalist['target'].tolist().count(elem) > 2 for elem in datalist['target'].tolist()):
    #     testdatalist = datalist.sample(frac=0.4, replace=False)

        
    # print(traindatalist)
    # print(testdatalist)

    
    # temp = []
    # rowsSelected = []

    # print(datalist)
    # for i in range(len(datalist)):
    #     print(i)
    #     print(datalist['target'][i])
    #     if datalist['target'][i] not in temp:
    #         temp.append(datalist['target'][i])
    #         rowsSelected.append(i)
    #         print('added')
    #     print('not added. useless.')

    # # rowsNotSelected = [i for i in list(range(len(datalist))) if i not in rowsSelected]
    # temp = np.random.choice(temp, round(len(datalist)*0.5), replace=False)
    
    # rowsSelected = rowsSelected.extend(temp)
    # # rowsNotSelected = [i for i in list(range(len(datalist))) if i not in rowsSelected]
            
    # traindatalist = datalist.iloc[rowsSelected]
    # print(len(traindatalist))
    # testdatalist = datalist.drop(index=rowsSelected)


    # print(len(testdatalist))
    # print(traindatalist)
    # print(testdatalist)
    # print(datalist)
    # print(rowsSelected)


    # print("train data lsit")
    # print(traindatalist)
    # print(traindatalist['data'].to_numpy())

    # print("actual dataframe generated")
    # print(pd.DataFrame(traindatalist['data'].to_numpy()))

    # print('train columns')
    # print(traindatalist.columns)

    
    print('training...')
    cl, le = sktrain(traindatalist, vc=cl, le=le)

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

def chatbot(statement):
    global cl
    global le
    global names
    global nlp
    
    statement = nlp(statement)
    print(statement)


    likelihoods = dict(map(lambda a: (a, max([statement.similarity(x) for x in queries.get(a).examples])), queries))
    print(likelihoods)
    largest = max(likelihoods, key=likelihoods.get)
    print(largest)
    queries.get(largest).response()

    print("sklearn")
    print(le.inverse_transform(cl.predict([statement.vector])))
    print("sklearn probabilities")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = cl.predict_proba([statement.vector]).tolist()[0]
            print(prob)
            print(le.inverse_transform(range(len(prob))))
            print(dict(zip(le.inverse_transform(range(len(prob))), prob)))
    except:
        print("probabilities not available")
    
    queries.get(le.inverse_transform(cl.predict([statement.vector]))[0]).response()
    
def run():
    tempvar = ""
    leaverlist = queries['farewell'].examples
    leaverlist = leaverlist + [nlp(x) for x in ["quit.", "quit", "exit", "leave", "goodbye"]]
    while max([nlp(tempvar).similarity(leaver) for leaver in leaverlist]) < 0.85:
        tempvar = input("say something!")
        print()
        chatbot(tempvar)
        print()
        print("leaver max:")
        print(max([nlp(tempvar).similarity(leaver) for leaver in leaverlist]))
        print()
        print()

failedtrain = pd.DataFrame(columns=['data','target'])
failedtest = pd.DataFrame(columns=['data','target'])


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    score = 0
    classifier = getModel()
    print('testing model')
    while score < threshold:
        score, currtrain, failedtest, currdatalist = sktestQuery(queries, cl=classifier)
        testdatalist = pd.DataFrame(columns=['data','target'])
        


cl, le = sktrainQuery(queries)
print(failedtrain)
print(failedtest)
print(le.inverse_transform(failedtest['target']))

print('scoring final classifier on last test dataset')
score = cl.score(np.stack([d[0] for d in failedtest['data'].values.tolist()]), failedtest['target'])
print(score)

print('scoring final classifier on random portions of dataset')
for i in range(10):
    while not all(testdatalist['target'].tolist().count(elem) > 2 for elem in currdatalist['target'].tolist()):
        testdatalist = currdatalist.sample(frac=0.4, replace=False)
    testdatalist['target'] = le.transform(testdatalist['target'])
    score = cl.score(np.stack([d[0] for d in testdatalist['data'].values.tolist()]), testdatalist['target'])
    print(score)


print('testing classifier design on default (non-text) datasets')
print('NOT IMPLEMENTED')

print()

run()
