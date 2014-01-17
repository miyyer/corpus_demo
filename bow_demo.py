import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import itertools
 
# generate bag-of-words features from raw text
def word_feats(raw):

    # tokenize text into list of words
    words = nltk.wordpunct_tokenize(raw)

    # map to boolean dictionary (e.g. 'good' --> True, 'bad' --> True)
    return dict([(word, True) for word in words])

# generate bigram features scored by chi-squared
def bigram_feats(raw, score_fn=BigramAssocMeasures.chi_sq, n=200):

    # tokenize, as before
    words = nltk.wordpunct_tokenize(raw)

    # get all bigrams that occur in the text
    bigram_finder = BigramCollocationFinder.from_words(words)

    # score these bigrams
    best_bigrams = bigram_finder.nbest(score_fn, n)

    # return boolean mapping, as before
    return dict([(ngram, True) for ngram in itertools.chain(words, best_bigrams)])

if __name__ == '__main__':
 

    # load corpus, will be different for corpora not included in nltk
    neg_ids = movie_reviews.fileids('neg')
    pos_ids = movie_reviews.fileids('pos')
     
    # bag-of-words features
    neg_feats = [(word_feats(movie_reviews.raw(fileids=[f])), 'neg') for f in neg_ids]
    pos_feats = [(word_feats(movie_reviews.raw(fileids=[f])), 'pos') for f in pos_ids]

    # uncomment for bag-of-bigram feats (make sure to comment out the preceding two lines)
    # neg_feats = [(bigram_feats(movie_reviews.raw(fileids=[f])), 'neg') for f in neg_ids]
    # pos_feats = [(bigram_feats(movie_reviews.raw(fileids=[f])), 'pos') for f in pos_ids]
     
    # create train / test split
    negcutoff = len(neg_feats)*3/4
    poscutoff = len(pos_feats)*3/4

    train_feats = neg_feats[:negcutoff] + pos_feats[:poscutoff]
    test_feats = neg_feats[negcutoff:] + pos_feats[poscutoff:]
    print 'train on %d instances, test on %d instances' % (len(train_feats), len(test_feats))
     
    # classify
    classifier = NaiveBayesClassifier.train(train_feats)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_feats)

    # show accuracy
    classifier.show_most_informative_features(n=30)