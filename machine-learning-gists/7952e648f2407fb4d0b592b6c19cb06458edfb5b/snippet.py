from gensim.models.word2vec import Word2Vec
from gensim import corpora, models, similarities
from gensim.models.phrases import Phrases
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import re
import codecs
import unicodecsv as csv
from nltk import classify, tokenize
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
from HTMLParser import HTMLParser
parser = HTMLParser()
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
import nltk

from nltk.corpus import stopwords # Import the stop word list
myStopWords = stopwords.words("english")

# Add some custom stopwords from the data
for stop in (u'', u'&amp;', u'null'):
    myStopWords.append(stop)
myStopWords = set(myStopWords)


def cleanText(corpus):
	print("Cleaning %d lines." % len(corpus))
	index = 0
	cleaned = []
	for line in corpus:
		index += 1
		if index%10000 == 0:
			print(" Line: %d" % index)
		line = parser.unescape(line.lower())
		line = [word for sentence in [tokenizer.tokenize(sent) for sent in tokenize.sent_tokenize(line)] for word in sentence] # Fast-ish if unreadable approach to tokenize sentences and words in one go.
		if len(line) >= 3:
			line[0] = (u'pid_' + line[0])
			line[1] = (u'forumId_' + line[1])
			line = [word for word in line if word not in myStopWords]
			cleaned.append(line)		
	return cleaned	

# Skips some invalid records and adds all the CSV collumns to the space 
# delimited set of rows. May not want all the rows depending on data
def csvToLines(filename):
	print("Reading %s" % filename)
	lines = []
	lineNum = 0
	index = {}
	with codecs.open(filename, 'r', 'utf-8', errors='ignore') as csvIN:
		outCSV=(line for line in csv.reader(csvIN))
		for row in outCSV:
				if len(row) > 2 and row[1] != '10' and row[0] != 'postId':
					#lines.append(u'pid_' + unicode.join(u' ', row))
					lines.append(unicode.join(u' ', row))
					index[lineNum] = row[0]
					lineNum += 1
	return (lines, index)

#CSV files 
(unclassifiedLines, uIndex) = csvToLines("unclassified.csv")
(bad_sentiment_lines, bIndex) = csvToLines("good.csv")
(good_sentiment_lines, okIndex) = csvToLines("bad.csv")

unclassifiedLines = cleanText(unclassifiedLines)
bad_sentiment_lines = cleanText(bad_sentiment_lines)
good_sentiment_lines = cleanText(good_sentiment_lines)

all = unclassifiedLines + bad_sentiment_lines + good_sentiment_lines

dictionary = corpora.Dictionary(all)
dictionary.filter_extremes(no_below=2, no_above=0.5) #Remove words that appear in less than 2 documents and more than half of them.


"""
Print top 50 words that are left to double check
that most irrelevant terms are removed
"""
from collections import defaultdict
frequency = defaultdict(int)
for doc in all:
	for word in doc:
		if word in dictionary.token2id:
			frequency[word] += 1
token_order = [(k, frequency[k]) for k in sorted(frequency, key=frequency.get, reverse=True)]

for i in range(50):
		print("%d %s" % (i, token_order[i]))


#dictionary.save('myModel.dict')

unclassified2bow = [dictionary.doc2bow(line) for line in unclassifiedLines]
#corpora.MmCorpus.serialize('unclassified2bow.mm', unclassified2bow)

bad_sentiment_lines2bow = [dictionary.doc2bow(line) for line in bad_sentiment_lines]

#corpora.MmCorpus.serialize('bad_sentiment_lines2bow.mm', bad_sentiment_lines2bow)

good_sentiment_lines2bow = [dictionary.doc2bow(line) for line in good_sentiment_lines]
#corpora.MmCorpus.serialize('good_sentiment_lines2bow.mm', good_sentiment_lines2bow)

"""
# How to load data from a previous run if you wanted
dictionary = corpora.Dictionary.load('myModel.dict')
unclassified2bow = corpora.MmCorpus('unclassified2bow.mm')
bad_sentiment_lines2bow = corpora.MmCorpus('bad_sentiment_lines2bow.mm')
good_sentiment_lines2bow = corpora.MmCorpus('good_sentiment_lines2bow.mm')
"""

# Bag-Of-Words to Dictionary
def bow2dict(bow):
	features = {}
	for (k,v) in bow:
		features[dictionary[k]] = v
	return features
	
# Bag-Of-Words + classification to "featureset" data structure.
def featureSet(bows, classification):
	fset = []
	for bow in bows:
		fset.append((bow2dict(bow), classification))
	return fset

# split known data into training and testing sets.
# 1/4 of the data is held out for testing predictions.
train_set =  featureSet(bad_sentiment_lines2bow[len(bad_sentiment_lines2bow)/4:], 'negative')
train_set += featureSet(good_sentiment_lines2bow[len(good_sentiment_lines2bow)/4:], 'positive')
test_set = featureSet(bad_sentiment_lines2bow[:len(bad_sentiment_lines2bow)/4], 'negative')
test_set += featureSet(good_sentiment_lines2bow[:len(good_sentiment_lines2bow)/4], 'positive')

#fast, good baseline
from nltk import NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(100)
print classify.accuracy(nb_classifier, test_set)
#0.916118249243

def classifyStr(string, classifier):
	words = string.split()
	d = bow2dict(dictionary.doc2bow(words))
	return  classifier.prob_classify(d).prob('negative')
	
def explain(string, classifier):
	words = string.split()
	d = bow2dict(dictionary.doc2bow(words))
	return  classifier.explain(d)

# Write out 
def writeNegative(classifier, filename, sklearn_classifier = False):
	print("writing Negative %s" % filename)
	classification = []
	for i in range(len(unclassified2bow)):
		d = bow2dict(unclassified2bow[i])
		
		if sklearn_classifier:
			cl = classifier.classify(d)
		else:
			cl = classifier.prob_classify(d).prob('negative')
				
		classification.append((i, uIndex[i], cl, unclassifiedLines[i]))
	
	if sklearn_classifier:	
		classification.sort(key=lambda tup: tup[2]) #order by negative likelyhood rank
	else:
		classification.sort(key=lambda tup: -tup[2])
	
	url = 'https://www.example.com/en/Post/'
	Negative = []
	Negative.append(('row', 'url', 'score', 'feed','text'))
	for c in classification:
		(i, pid, cl, raw) = c
		Negative.append((i, url + str(pid), str(cl), raw[1], unicode.join(u' ', raw[2:])))


	with open(filename, 'wb') as csvfile:
		wr = csv.writer(csvfile, delimiter=',')
		for s in Negative:
			wr.writerow(s)		


#writeNegative(nb_classifier, "nbNegative.csv")

#--- Best one, but sort of slow.
"""
from nltk import ConditionalExponentialClassifier
ce_classifier = ConditionalExponentialClassifier.train(train_set, max_iter=10, max_acc=0.99)
print classify.accuracy(ce_classifier, test_set)
writeNegative(ce_classifier, "ceNegative.csv")
ce_classifier.show_most_informative_features(100)
"""