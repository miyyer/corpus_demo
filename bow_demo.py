import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import itertools, sys
 
def word_feats(raw):
    words = nltk.wordpunct_tokenize(raw)
    return dict([(word, True) for word in words])

def bigram_feats(raw, score_fn=BigramAssocMeasures.chi_sq, n=200):
    words = nltk.wordpunct_tokenize(raw)
    bigram_finder = BigramCollocationFinder.from_words(words)
    best_bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, best_bigrams)])

if __name__ == '__main__':
 
    neg_ids = movie_reviews.fileids('neg')
    pos_ids = movie_reviews.fileids('pos')
     
    neg_feats = [(bigram_feats(movie_reviews.raw(fileids=[f])), 'neg') for f in neg_ids]
    pos_feats = [(bigram_feats(movie_reviews.raw(fileids=[f])), 'pos') for f in pos_ids]

    print 'neg feats = ', len(neg_feats)
    print 'pos feats = ', len(pos_feats)
     
    negcutoff = len(neg_feats)*3/4
    poscutoff = len(pos_feats)*3/4


    train_feats = neg_feats[:negcutoff] + pos_feats[:poscutoff]
    test_feats = neg_feats[negcutoff:] + pos_feats[poscutoff:]
    print 'train on %d instances, test on %d instances' % (len(train_feats), len(test_feats))
     
    classifier = NaiveBayesClassifier.train(train_feats)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_feats)

    classifier.show_most_informative_features(n=30)