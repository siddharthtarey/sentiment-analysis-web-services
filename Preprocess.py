from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import twitter_samples

import csv
import nltk
import string

class preProcess:

    # remove certain stop words from the dictionary.
    def stopWordGenerate(self):
        # get stopwords from nltk corpus
        s = set(stopwords.words("english"))
        # remove stop words which could contribute to a negative or positive sentiment
        s.remove("no")
        s.remove("not")
        s.remove("don't")
        s.remove("against")
        s.remove("didn")
        s.remove("doesn")
        s.remove("needn't")
        s.remove("needn")
        s.remove("isn")
        s.remove("ain")
        s.remove("weren")
        s.remove("shan")

        return s

    # remove stop words from the docuument
    def stopWordRemoval(self,document,stopWords):
        wordTokens = word_tokenize(document)
        sentence = []

        for w in wordTokens:
            if(w not in stopWords):
                sentence.append(w)
        return sentence

    # performs word stemming for every document
    def wordStemming(self,sentence):
        porter = PorterStemmer()
        stemmedStentence = ""
        for word in sentence:
            stemmedStentence += " "+porter.stem(word)

        return stemmedStentence.strip()

    # generate bigrams.
    def generateBigrams(self,document):

        bigrams = list(nltk.bigrams(document))

        return bigrams

    def extractFromCsv(self,filename):
        f = open(filename, 'rt')
        reader = csv.reader(f)
        next(reader, None)
        data =[]
        for row in reader:
            data.append([row[0],row[1]])

        return data

    def main(self):
        print("Preprocess begin")
        stopWords = self.stopWordGenerate()

        data1 = twitter_samples.strings('negative_tweets.json')
        data2 = twitter_samples.strings('positive_tweets.json')
        data = data1+data2

        stemmedList = []
        score = []
        # iterate through every document in the corpus.
        for document in data:
            doc = document[0].lower()
            doc = doc.translate(str.maketrans('','',string.punctuation))
            cleanSentence = self.stopWordRemoval(doc, stopWords)
            stemmedSentence = self.wordStemming(cleanSentence)
            stemmedList.append(stemmedSentence)
            score.append(int(document[1]))

        return stemmedList,score



