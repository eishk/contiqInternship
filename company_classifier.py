import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
from sqlalchemy import *
import csv

class ConceptClassifier():

    vocab = set()

    wordlist = []
    target = []
    weights = []

    def __init__(self, pickle_file):

        if pickle_file:
            self.load_model(pickle_file)

    def load_model(self,pickle_file):

        pkl_file = open(pickle_file, 'rb')
        self.model = pickle.load(pkl_file)
        self.wordlist = pickle.load(pkl_file)
        self.vocab = pickle.load(pkl_file)
        pkl_file.close()

    def load_data(self,db,concepts):
        
        pages = []
        with open('/Users/contiq/Documents/ok/finalcsv4.csv', 'rb') as csvfile:
            e_writer = csv.reader(csvfile, delimiter= '')
            for row in e_writer:
                page =[row[0], row[1], row[2], row[3], row[4]]
                pages.append(page)

        self.training = pages

    def clean_tokenize(self,text):

        # 0. Replace " with the word QuoteSign
        print (text)
        if text.find('"')>0:
            text = text.replace('"',' QuoteSign ')
            #print text

        # 1. Remove urls
        text = re.sub(r"(?:\@|https?\://)\S+", "", text, flags=re.MULTILINE)

        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", text)

        # 3. Convert to lower case, split into individual words
        words = letters_only.lower()#.split()

        # 4. Stemming
        stems = words.split()
        #stems = self.stem_tokenize(words)

        return stems #words

    def text_to_words(self, text, build=True):
        """
        Convert a raw text to a string of words.
        Input: a single string of words
        Output: a single string of preprocessed text
        """

        words = self.clean_tokenize(text)

        # Remove stop words
        stops = [] #set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        if build:
            for mw in meaningful_words:
                self.vocab.add(mw)
        else:
            dd = []
            for mw in meaningful_words:
                if mw in self.vocab:
                    dd.append(mw)
            meaningful_words = dd
        # Return a list of words into a string separated by space.
        return (" ".join(meaningful_words))

    def classify(self, text, type):
        if text.strip():
            vect = TfidfVectorizer(analyzer="word", stop_words='english', max_features=50000)
            text = self.text_to_words(text, build=False)
            vect.fit_transform(self.wordlist)
            #print text
            pred_doc = vect.transform([text])
            #print pred_doc
            pred = self.model.predict(pred_doc)
            classes = self.model.classes_
            pred_prob = self.model.predict_proba(pred_doc)
            prob = zip(classes,pred_prob[0])
            #print pred[0],pred_prob
            #pred_prob = pred_prob[0]
            if type==1:
                return str(pred[0])
            else:
                return prob
        else:
            return "0"

    def enrich_text(self, text, concept, page):
        if text.count('$')>0:
            print( '**********:',concept,text.count('$'))
        if len(text) < 200:
            len0 = " texlenOne"
        elif len(text) < 500:
            len0 = " texlenTwo"
        elif len(text) < 1000:
            len0 = " texlenThree"
        elif len(text) < 2000:
            len0 = " texlenFour"
        else:
            len0 = " texlenFive"
        tex = text + len0 + ' ' + concept
        if page == 1:
            ret = self.text_to_words(tex + " FirstPage")
        elif page == 2:
            ret = self.text_to_words(tex + " SecondPage")
        else:
            ret = self.text_to_words(tex)
        return ret

    def build_classifier(self):

        for tex in self.training:
            self.wordlist.append(self.enrich_text(tex[1],tex[2],tex[3]))
            self.target.append(tex[2])

        vect = TfidfVectorizer(analyzer="word", stop_words='english', max_features=50000)
        #        vect = CountVectorizer(analyzer="word", stop_words='english', max_features=500000)

        X = vect.fit_transform(self.wordlist)
        y = self.target

        print( vect.get_feature_names())
        print( len(vect.get_feature_names()))
        feature_names = vect.get_feature_names()

        # CROSS VALIDATION
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

        print("crossvalidation")
        print( len(y_train), len(y_test))
        print( y_train)
        vc = LogisticRegression(class_weight="balanced", penalty='l2', C=10)
        shuffle = KFold(len(y_train), n_folds=10, shuffle=True, random_state=None)
        scores = cross_val_score(vc, X_train, y_train, cv=shuffle)
        print(scores)
        print(np.mean(scores))

        # vc = SVC(kernel='linear', probability=True)
        vc.fit(X_train, y_train)
        print( "after fit")
        score = vc.score(X_train, y_train)
        print( 'train score: ', score)
        y_pred = vc.predict(X_train)
        print( confusion_matrix(y_train, y_pred))

        score = vc.score(X_test, y_test)
        print( 'test score: ', score)
        y_pred = vc.predict(X_test)
        print (confusion_matrix(y_test, y_pred))

        self.model = vc
        for i in range(0, vc.coef_.shape[0]):
            # coef = vc.coef_[i][0].toarray()[0] #SVC
            coef = vc.coef_[i]  # SGD & #LR
            top20_indices = np.argsort(coef)[-30:]
            top20 = np.sort(coef)
            print( vc.classes_[i])
            for j in range(1, 31):
                print( '\t', j, ': ', feature_names[top20_indices[-j]], top20[-j])
        return vc

    def find_main_class(self, text, page):
        if text.strip():
            vect = TfidfVectorizer(analyzer="word", stop_words='english', max_features=50000)
            text = self.text_to_words(text, build=False)
            vect.fit_transform(self.wordlist)
            # print text
            tex = self.enrich_text(text, "", page)
            pred_doc = vect.transform([tex])
            # print pred_doc
            pred = self.model.predict(pred_doc)
            classes = self.model.classes_
            pred_prob = self.model.predict_proba(pred_doc)
            preds = pred_prob[0]
            score = max(preds)
            print( score,preds)
            # pred_prob = pred_prob[0]
            if score > .4:
                i = np.argmax(preds)
                return classes[i] + " - " + str(int(score*100)) + '%'
            else:
                return ""
        else:
            return ""

    def save_model(self,pickle_file):

        output = open(pickle_file, 'wb')
        pickle.dump(self.model, output)
        pickle.dump(self.wordlist, output)
        pickle.dump(self.vocab, output)
        output.close()

def build_model(pkl_name):

    # read training data from file
    db = ""
    cc = ConceptClassifier(None)
    
    concepts = "'company','product','other'"
    cc.load_data(db,concepts)
    cc.build_classifier()
    cc.save_model(pkl_name)


def test_model(pkl_name):
    text = 'EPG | MICROSOFT CONFIDENTIAL Qualify/Develop: Competition 13 Microsoft + Yammer combines two Gartner'
    cc = ConceptClassifier(pkl_name)
    res = cc.find_main_class(text,13)
    print( '*',res, '*')


if __name__ == "__main__":
    pkl_name = "model15.pkl"
    build_model(pkl_name)
    #test_model(pkl_name)
