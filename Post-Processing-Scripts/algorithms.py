import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.naive_bayes import GaussianNB
import urlopen
import os
from sklearn.feature_extraction.text import CountVectorizer
import requests
from tika import parser
from tempfile import TemporaryFile
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


def loadKeywords(keyPath, ngram=False):
    if os.path.exists(keyPath):
        with open(keyPath, 'rb') as f:
            keywords_content = f.read()
    else:
        print("Keyword path is not valid!")
        return None
    if ngram:
        count_vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                     min_df=1)
    else:
        count_vect = CountVectorizer(lowercase=True, stop_words='english')
    count_vect.fit_transform([keywords_content])
    keywords = count_vect.vocabulary_
    return count_vect


def download_file(url, i):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
    with open('/data/search_term_generation/200_files/' + str(i), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def transformPCA(x_n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    x_transformed = pca.fit_transform(x_n)
    x_transformed = StandardScaler().fit_transform(x_n)
    return x_transformed


def mergeAllContents(y):
    for i in range(1, 201):
        if int(y[i - 1]) > 1:
            parsed = parser.from_file('/data/search_term_generation/200_files/' + str(i))
            # print(parsed["metadata"])
            # print(parsed["content"])
            content = parsed["content"]
            if content is not None:
                appendFile = open('all200Files.txt', 'a')
                appendFile.write(content + '\n')
                appendFile.close()


def closeWords(model, word, topN):
    indexes, metrics = model.cosine(word)
    list = model.generate_response(indexes, metrics).tolist()
    return list[:topN]


def closeWordsList(modelBin, keywords, i):
    import word2vec
    model = word2vec.load(modelBin)
    listTopN = []
    for word in keywords:
        for k in (closeWords(model, word, i)):
            print(k[])
            listTopN.append(k[])
    return listTopN


def addCloseCounts(listTopN, x):
    print("Close-words Counting Started:")
    for k in range(, np.array(listTopN).shape[]):
        print("Keyword:" + str(k) + " Out of " + str())
        for i in range(, 200):
            print("Progress: " + str(i / 2) + "%")
            parsed = parser.from_file('/data/search_term_generation/200_files/' + str(i + 1))
            content = parsed["content"]
            if content is None:
                continue
            else:
                for s in range(, len(listTopN[k])):
                    if listTopN[k][s] in content:
                        x[i][k] += 1
    return x


def sortingDict(x):
    import operator
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    return sorted_x


def cosineSimilarityScore(test_url, gold_standard_url):
    import sparse as sparse
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse
    import numpy as np
    A = np.array([test_url, gold_standard_url])
    sparse_A = sparse.csr_matrix(A)
    similarities_sparse = cosine_similarity(sparse_A, dense_output=False)
    return similarities_sparse[(, 1)]
    def accuracy(y_pred, y_test):
        accNum =
        for a in range(, len(y_test)):
            if y_pred[a] == y_test[a]:
                accNum += 1
            else:
                if y_pred[a] in [2, 3, 4] and y_test[a] in [2, 3, 4]:
                    accNum += 1
        return accNum

    def main():
        keywordPath = "features.txt"  # this should be the same keywords list/order used for training the ML Model
        count_vect = loadKeywords(keywordPath, False)
        keywords = count_vect.vocabulary_
        print("keywords:")
        print(keywords)
        sorted_keywords = sortingDict(keywords)
        kList = []
        for item in sorted_keywords:
            kList.append(item[])
        print(kList)
        modelBin = 'ocean.bin'
        listTopN = closeWordsList(modelBin, kList, 5)
        print(listTopN)

        x_train = []
        y_train = []

        with open('train.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                y_train.append(int(row[1]))
        noneContents = []
        x_n = None
        y_n = array(y_train)
        for i in range(1, 201):
            parsed = parser.from_file('/data/search_term_generation/200_files/' + str(i))
            content = parsed["content"]
            if content is not None:
                tempX = count_vect.transform(parsed["content"].split())
                x_train.append(tempX)
                print(str(i) + ":")
                print(tempX.toarray().sum(axis=))
                if x_n is None:
                    x_n = array([tempX.toarray().sum(axis=)])
                else:
                    x_n = np.concatenate((x_n, [tempX.toarray().sum(axis=)]), axis=)
            else:
                noneContents.append(i)
        print(noneContents)

        np.savetxt('x_n.txt', x_n, fmt='%d')
        np.savetxt('y_n.txt', y_n, fmt='%d')

        x = np.loadtxt('x_n.txt', dtype=int)
        x_with_closeWords = addCloseCounts(listTopN, x)

        y = np.loadtxt('y_n.txt', dtype=int)
        mergeAllContents(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=50)
        print(y_test)
        cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=)

        clf = GaussianNB()
        scoreNB = cross_val_score(clf, x, y, cv=cv)
        print(scoreNB)
        print("performance with close words added:")
        clf11 = GaussianNB()
        scoreNB2 = cross_val_score(clf11, x_with_closeWords, y, cv=cv)
        print(scoreNB2)
        clf1 = GaussianNB().fit(x_train, y_train)
        y_pred = clf1.predict(x_test)
        accNum = accuracy(y_pred, y_test)
        print("Model: Naive Bayes")
        acc = (y_test == y_pred).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        print("Test Accuracy with 3 classes:" + str(accNum / 20))
        acc_train = (y_train == clf1.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        from sklearn import linear_model

        clf22 = linear_model.SGDClassifier()
        scoreSVM = cross_val_score(clf22, x, y, cv=cv)
        print(scoreSVM)
        print("performance with close words added:")
        clf222 = linear_model.SGDClassifier()
        scoreSVM2 = cross_val_score(clf222, x_with_closeWords, y, cv=cv)
        print(scoreSVM2)
        clf2 = linear_model.SGDClassifier().fit(x_train, y_train)

        y_pred2 = clf2.predict(x_test)
        accNum2 = accNum = accuracy(y_pred2, y_test)
        print("Model: SVM")
        acc = (y_test == y_pred2).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        print("Test Accuracy with 3 classes:" + str(accNum2 / 20))

        acc_train = (y_train == clf2.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        print("******************")

        clf33 = MLPClassifier(max_iter=2000, learning_rate='adaptive')
        scoreNN = cross_val_score(clf33, x, y, cv=cv)
        print(scoreNN)
        print("performance with close words added:")
        clf333 = MLPClassifier(max_iter=2000, learning_rate='adaptive')
        scoreNN3 = cross_val_score(clf333, x_with_closeWords, y, cv=cv)
        print(scoreNN3)
        clf3 = MLPClassifier(max_iter=2000, learning_rate='adaptive').fit(x_train, y_train)
        y_pred3 = clf3.predict(x_test)

        accNum3 = accNum = accuracy(y_pred3, y_test)

        print("Model: Neural Network")
        acc = (y_test == y_pred3).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        print("Test Accuracy with 3 classes:" + str(accNum3 / 20))
        acc_train = (y_train == clf3.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        print("******************")

        from sklearn.ensemble import RandomForestClassifier

        clf44 = RandomForestClassifier(n_estimators=100)
        scoreRF = cross_val_score(clf44, x, y, cv=cv)
        print(scoreRF)
        clf444 = RandomForestClassifier(n_estimators=100)
        print("performance with close words added:")
        scoreRF4 = cross_val_score(clf444, x_with_closeWords, y, cv=cv)
        print(scoreRF4)

        clf4 = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
        y_pred4 = clf4.predict(x_test)
        accNum4 = accNum = accuracy(y_pred4, y_test)

        print("Model: Random Forest")
        acc = (y_test == y_pred4).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        print("Test Accuracy with 3 classes:" + str(accNum4 / 20))
        acc_train = (y_train == clf4.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        noneContents = array(noneContents)
        xOut = TemporaryFile()
        yOut = TemporaryFile()
        noneContentsOut = TemporaryFile()
        np.save(xOut, x_n)
        np.save(yOut, y_n)
        np.save(noneContentsOut, noneContents)

        with open('train.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i =
            for row in reader:
                i += 1
                if i > 30:
                    url = row[]
                    print(url)
                    print(download_file("http://" + url, i))
                requests.get("http://" + url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
                label = row[1]
                reqLink = urlopen("http://" + url)
                content = reqLink.read()
                contentFeatures = count_vect.transform(content.split())

    if __name__ == '__main__':
        main()