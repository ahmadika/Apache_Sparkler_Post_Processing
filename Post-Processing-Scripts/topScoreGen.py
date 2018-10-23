#This code is for post-processing, insights/data Analysis related to crawled data by Sparkler
#User needs to provide the links to Solr cores, ML Models, and utilized keywords lists
#This code is developed by Simin Ahmadi Karvigh and Omid Davtalab in Summer 2018 at USC IRDS Lab
#For questions please contact davtalab@usc.edu

from urllib import urlopen
#import pickle
import os
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib #NOT THERE
#import requests
import numpy as np
#import pysolr
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import json

def loadKeywords(keyPath):
    if os.path.exists(keyPath):
        with open(keyPath, 'rb') as f:
            keywords_content = f.read()
    else:
        print("Keyword path is not valid!")
        return None
    count_vect = CountVectorizer(lowercase=True, stop_words='english')
    count_vect.fit_transform([keywords_content])
    keywords = count_vect.vocabulary_
    return count_vect

def loadDocUrls(url):
    #Call this function using the url to Solr core:
    #e.g. loadDocUrls("http://localhost:8983/solr/crawldb")
    #maxRows = 2147483647 #this is the maximum # of documents in Solr
    #maxRows=5  #change maxRows if you need specific number of docs
    rows=264794 #current number of fetched docs
    url+="/select?fq=status:FETCHED&indent=on&q=*:*&rows="+str(rows)+"&wt=json"
    #url += "/select?status:FETCHED&indent=on&q=*:*&rows="+str(rows)+"&start="+str(start)+"&wt=json"
    connection = urlopen(url)
    response = json.load(connection)
    docs = response['response']['docs']
    numDocs = response['response']['numFound']  # total number of crawled documents
    print ("Total Number of Fetched Documents in Solr:", numDocs)
    return docs, numDocs

def totalCrawledDocs(url):
    url += "/select?indent=on&q=*:*&rows=5&wt=python"
    connection = urlopen(url)
    response = eval(connection.read())
    numDocs = response['response']['numFound']  # total number of crawled documents
    return numDocs

def loadModel(pathToModel):
    if os.path.exists(pathToModel):
        print("Loading the Model...")
        #loaded_model = pickle.load(open(pathToModel, 'rb'))
        loaded_model =joblib.load(pathToModel)
        return loaded_model
    else:
        print("No Model Exists")
        return None

def updateSolrScore(url,id, score):
    solr = pysolr.Solr(url)
    doc = {'id':str(id), 'page_score':score}
    solr.add([doc], fieldUpdates={'page_score':'set'})

def loadNBmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = GaussianNB().fit(x, y)
    joblib.dump(clf, 'NB_model.pkl', protocol=2)
    return clf

def loadSVMmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    #clf = linear_model.SGDClassifier(max_iter=1000,loss='log').fit(x, y)
    clf = linear_model.SGDClassifier(loss='log').fit(x, y)
    joblib.dump(clf, 'SVM_model.pkl', protocol=2)
    return clf

def loadNNmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = MLPClassifier(max_iter=2000, learning_rate='adaptive').fit(x, y)
    joblib.dump(clf, 'NN_model.pkl', protocol=2)
    return clf

def loadRFmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = RandomForestClassifier(n_estimators=100).fit(x, y)
    joblib.dump(clf, 'RF_model.pkl', protocol=2)
    return clf

def writeTotxt(filename, dict):
    if dict is None:
        print("Error: Dictionary is None!")
        return
    with open(filename, 'w') as file:
        file.write(json.dumps(dict))

def retSortedScore(keywordPath,solrLink,clfs,modelNames, rowsTokeep):
    count_vect = loadKeywords(keywordPath)
    keywords = count_vect.vocabulary_
    docs, numDocs = loadDocUrls(solrLink)
    solrDict1 = {}
    solrDict2 = {}
    solrDict3 = {}
    solrDict4 = {}
    for doc in docs:
        if 'extracted_text' in doc.keys():
            #counter+=1
            #print (counter)
            content = (doc['extracted_text'])
            content = content.lower()
            contentFeatures = count_vect.transform(content.split())
            featureVector=[contentFeatures.toarray().sum(axis=0)]
            if (np.sum(featureVector, axis=1))<2:
                solrDict1[doc['id']] = (doc['url'], 0.0)
                solrDict2[doc['id']] = (doc['url'], 0.0)
                solrDict3[doc['id']] = (doc['url'], 0.0)
                solrDict4[doc['id']] = (doc['url'], 0.0)
            else:
                probList1=clfs[0].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score1=probList1[0][1] + 2* probList1[0][2] + 3* probList1[0][3] + 4* probList1[0][4]- probList1[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict1[doc['id']]=(doc['url'],score1)

                probList2=clfs[1].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score2=probList2[0][1] + 2* probList2[0][2] + 3* probList2[0][3] + 4* probList2[0][4]- probList2[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict2[doc['id']]=(doc['url'],score2)

                probList3=clfs[2].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score3=probList3[0][1] + 2* probList3[0][2] + 3* probList3[0][3] + 4* probList3[0][4]- probList3[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict3[doc['id']]=(doc['url'],score3)

                probList4=clfs[3].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score4=probList4[0][1] + 2* probList4[0][2] + 3* probList4[0][3] + 4* probList4[0][4]- probList4[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict4[doc['id']]=(doc['url'],score4)


    print(len(solrDict1))
    rowsTokeep=min(rowsTokeep,len(solrDict1))
    sorted_scores=sorted(solrDict1.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[0]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict2.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[1]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict3.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[2]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict4.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[3]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

keywordPath="/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/features.txt" #this should be the same keywords list/order used for training the ML Model
solrLink="http://localhost:8983/solr/crawldb"
#x_path = 'data/x_n.txt'
#y_path = 'data/y_n.txt'
x_path = '/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/x_n.txt'
y_path = '/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/y_n.txt'

clfs=[]


nbModel = loadModel('NB_model.pkl')
if nbModel is None:
    nbModel = loadNBmodel(x_path, y_path)
clfs.append(nbModel)
print("NB Model Classes:" + str(nbModel.classes_))
#retSortedScore(keywordPath,solrLink,nbModel,"NB",250)

SVMModel = loadModel('SVM_model.pkl')
if SVMModel is None:
    SVMModel = loadSVMmodel(x_path, y_path)
print("SVM Model Classes:" + str(SVMModel.classes_))
#retSortedScore(keywordPath,solrLink,SVMModel,"SVM",250)
clfs.append(SVMModel)

NNModel = loadModel('NN_model.pkl')
if NNModel is None:
    NNModel = loadNNmodel(x_path, y_path)
print("NN Model Classes:" + str(NNModel.classes_))
#retSortedScore(keywordPath,solrLink,NNModel,"NN_Fixed",250)
clfs.append(NNModel)

RFModel = loadModel('RF_model.pkl')#This code is for post-processing, insights/data Analysis related to crawled data by Sparkler
#User needs to provide the links to Solr cores, ML Models, and utilized keywords lists
#This code is developed by Simin Ahmadi Karvigh and Omid Davtalab in Summer 2018 at USC IRDS Lab
#For questions please contact davtalab@usc.edu

from urllib import urlopen
#import pickle
import os
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib #NOT THERE
#import requests
import numpy as np
#import pysolr
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import json

def loadKeywords(keyPath):
    if os.path.exists(keyPath):
        with open(keyPath, 'rb') as f:
            keywords_content = f.read()
    else:
        print("Keyword path is not valid!")
        return None
    count_vect = CountVectorizer(lowercase=True, stop_words='english')
    count_vect.fit_transform([keywords_content])
    keywords = count_vect.vocabulary_
    return count_vect

def loadDocUrls(url):
    #Call this function using the url to Solr core:
    #e.g. loadDocUrls("http://localhost:8983/solr/crawldb")
    #maxRows = 2147483647 #this is the maximum # of documents in Solr
    #maxRows=5  #change maxRows if you need specific number of docs
    rows=264794 #current number of fetched docs
    url+="/select?fq=status:FETCHED&indent=on&q=*:*&rows="+str(rows)+"&wt=json"
    #url += "/select?status:FETCHED&indent=on&q=*:*&rows="+str(rows)+"&start="+str(start)+"&wt=json"
    connection = urlopen(url)
    response = json.load(connection)
    docs = response['response']['docs']
    numDocs = response['response']['numFound']  # total number of crawled documents
    print ("Total Number of Fetched Documents in Solr:", numDocs)
    return docs, numDocs

def totalCrawledDocs(url):
    url += "/select?indent=on&q=*:*&rows=5&wt=python"
    connection = urlopen(url)
    response = eval(connection.read())
    numDocs = response['response']['numFound']  # total number of crawled documents
    return numDocs

def loadModel(pathToModel):
    if os.path.exists(pathToModel):
        print("Loading the Model...")
        #loaded_model = pickle.load(open(pathToModel, 'rb'))
        loaded_model =joblib.load(pathToModel)
        return loaded_model
    else:
        print("No Model Exists")
        return None

def updateSolrScore(url,id, score):
    solr = pysolr.Solr(url)
    doc = {'id':str(id), 'page_score':score}
    solr.add([doc], fieldUpdates={'page_score':'set'})

def loadNBmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = GaussianNB().fit(x, y)
    joblib.dump(clf, 'NB_model.pkl', protocol=2)
    return clf

def loadSVMmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    #clf = linear_model.SGDClassifier(max_iter=1000,loss='log').fit(x, y)
    clf = linear_model.SGDClassifier(loss='log').fit(x, y)
    joblib.dump(clf, 'SVM_model.pkl', protocol=2)
    return clf

def loadNNmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = MLPClassifier(max_iter=2000, learning_rate='adaptive').fit(x, y)
    joblib.dump(clf, 'NN_model.pkl', protocol=2)
    return clf

def loadRFmodel(x_path,y_path):
    x = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    clf = RandomForestClassifier(n_estimators=100).fit(x, y)
    joblib.dump(clf, 'RF_model.pkl', protocol=2)
    return clf

def writeTotxt(filename, dict):
    if dict is None:
        print("Error: Dictionary is None!")
        return
    with open(filename, 'w') as file:
        file.write(json.dumps(dict))

def retSortedScore(keywordPath,solrLink,clfs,modelNames, rowsTokeep):
    count_vect = loadKeywords(keywordPath)
    keywords = count_vect.vocabulary_
    docs, numDocs = loadDocUrls(solrLink)
    solrDict1 = {}
    solrDict2 = {}
    solrDict3 = {}
    solrDict4 = {}
    for doc in docs:
        if 'extracted_text' in doc.keys():
            #counter+=1
            #print (counter)
            content = (doc['extracted_text'])
            content = content.lower()
            contentFeatures = count_vect.transform(content.split())
            featureVector=[contentFeatures.toarray().sum(axis=0)]
            if (np.sum(featureVector, axis=1))<2:
                solrDict1[doc['id']] = (doc['url'], 0.0)
                solrDict2[doc['id']] = (doc['url'], 0.0)
                solrDict3[doc['id']] = (doc['url'], 0.0)
                solrDict4[doc['id']] = (doc['url'], 0.0)
            else:
                probList1=clfs[0].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score1=probList1[0][1] + 2* probList1[0][2] + 3* probList1[0][3] + 4* probList1[0][4]- probList1[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict1[doc['id']]=(doc['url'],score1)

                probList2=clfs[1].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score2=probList2[0][1] + 2* probList2[0][2] + 3* probList2[0][3] + 4* probList2[0][4]- probList2[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict2[doc['id']]=(doc['url'],score2)

                probList3=clfs[2].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score3=probList3[0][1] + 2* probList3[0][2] + 3* probList3[0][3] + 4* probList3[0][4]- probList3[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict3[doc['id']]=(doc['url'],score3)

                probList4=clfs[3].predict_proba(featureVector)
                # The score is weighted Sum of class 2 to 5 probs minus class one prob
                score4=probList4[0][1] + 2* probList4[0][2] + 3* probList4[0][3] + 4* probList4[0][4]- probList4[0][0]
                #print("Relevant Probability:"+ str(score)+'\n')
                solrDict4[doc['id']]=(doc['url'],score4)


    print(len(solrDict1))
    rowsTokeep=min(rowsTokeep,len(solrDict1))
    sorted_scores=sorted(solrDict1.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[0]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict2.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[1]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict3.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[2]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

    sorted_scores=sorted(solrDict4.items(), key=lambda x: x[1][1], reverse=True)[0:rowsTokeep]
    filename= modelNames[3]+"_C5.txt"
    writeTotxt(filename, sorted_scores)

keywordPath="/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/features.txt" #this should be the same keywords list/order used for training the ML Model
solrLink="http://localhost:8983/solr/crawldb"
#x_path = 'data/x_n.txt'
#y_path = 'data/y_n.txt'
x_path = '/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/x_n.txt'
y_path = '/data/projects/G-817549/sparkler/python-env/Apache_Sparkler_Post_Processing/Post-Processing-Scripts/data/y_n.txt'

clfs=[]


nbModel = loadModel('NB_model.pkl')
if nbModel is None:
    nbModel = loadNBmodel(x_path, y_path)
clfs.append(nbModel)
print("NB Model Classes:" + str(nbModel.classes_))
#retSortedScore(keywordPath,solrLink,nbModel,"NB",250)

SVMModel = loadModel('SVM_model.pkl')
if SVMModel is None:
    SVMModel = loadSVMmodel(x_path, y_path)
print("SVM Model Classes:" + str(SVMModel.classes_))
#retSortedScore(keywordPath,solrLink,SVMModel,"SVM",250)
clfs.append(SVMModel)

NNModel = loadModel('NN_model.pkl')
if NNModel is None:
    NNModel = loadNNmodel(x_path, y_path)
print("NN Model Classes:" + str(NNModel.classes_))
#retSortedScore(keywordPath,solrLink,NNModel,"NN_Fixed",250)
clfs.append(NNModel)

RFModel = loadModel('RF_model.pkl')
if RFModel is None:
    RFModel = loadRFmodel(x_path, y_path)
print("RF Model Classes:" + str(RFModel.classes_))
#retSortedScore(keywordPath,solrLink,RFModel,"RF",250)
clfs.append(RFModel)
modelNames=["NB","SVM","NN_Fixed","RF"]
retSortedScore(keywordPath,solrLink,clfs,modelNames,260000)
if RFModel is None:
    RFModel = loadRFmodel(x_path, y_path)
print("RF Model Classes:" + str(RFModel.classes_))
#retSortedScore(keywordPath,solrLink,RFModel,"RF",250)
clfs.append(RFModel)
modelNames=["NB","SVM","NN_Fixed","RF"]
retSortedScore(keywordPath,solrLink,clfs,modelNames,260000)