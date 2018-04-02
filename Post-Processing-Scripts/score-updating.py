#This code is for post-processing, insights/data Analysis related to crawled data by Sparkler
#User needs to provide the links to Solr cores, ML Models, and utilized keywords lists
#This code is developed by Simin Ahmadi Karvigh in Spring 2018 at USC IRDS Lab
#For questions please contact ahmadika@usc.edu

from urllib import urlopen
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
import requests
import pysolr

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
    maxRows = 2147483647 #this is the maximum # of documents in Solr
    #maxRows=5  #change maxRows if you need specific number of docs
    url += "/select?indent=on&q=*:*&rows="+str(maxRows)+"&wt=python"
    connection = urlopen(url)
    response = eval(connection.read())
    docs = response['response']['docs']
    numDocs = response['response']['numFound']  # total number of crawled documents
    print ("Total Number of crawled documents by Sparkler:", numDocs)
    return docs

def totalCrawledDocs(url):
    url += "/select?indent=on&q=*:*&rows=5&wt=python"
    connection = urlopen(url)
    response = eval(connection.read())
    numDocs = response['response']['numFound']  # total number of crawled documents
    return numDocs

def loadModel(pathToModel):
    if os.path.exists(pathToModel):
        print("Loading the Model...")
        loaded_model = pickle.load(open(pathToModel, 'rb'))
        return loaded_model
    else:
        print("No Model Exists")
        return None

def updateSolrScore(url,id, score):
    solr = pysolr.Solr(url)
    doc = {'id':str(id), 'page_score':score}
    solr.add([doc], fieldUpdates={'page_score':'set'})

solrLink="http://localhost:8983/solr/crawldb"
numDocs = totalCrawledDocs(solrLink)
#print ("Number of crawled documents by Sparkler:",numDocs)
docs = loadDocUrls(solrLink)

nbModelLink = "/models/NBModel.pkl"
nbModel = loadModel(nbModelLink)
keywordPath="/keywords/keywords.txt" #this should be the same keywords list/order used for training the ML Model
count_vect =loadKeywords(keywordPath)
keywords=count_vect.vocabulary_
print(keywords)
print(count_vect)
for doc in docs:
    print(doc['id'])
    reqLink = requests.get(doc['url'])
    content = reqLink.text
    content = content.lower()
    contentFeatures = count_vect.transform(content.split())
    print(count_vect.vocabulary_)
    print(contentFeatures.toarray().sum(axis=0))

    #This is the score for high relevancy class (1st class when training NB Model)
    #It should be replaced with page_score param in Solr schema of the copy core
    solrScore=nbModel.predict_proba(contentFeatures.toarray().sum(axis=0))[0]
    updateSolrScore(solrLink,doc['id'],solrScore)






