from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
import regex #pip install regex
from nltk.stem.snowball import SpanishStemmer
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression, SGDClassifier
import nltk



# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.
def word_shape(word):
    p = regex.compile(r"\p{Lu}")
    p1 = regex.compile(r"\p{Ll}")

    word = p.sub('X', word)
    word = p1.sub('x', word)
    word = regex.sub('[\d]', 'd', word)
    return word



def getfeats(fields, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    word = fields[0]
    stemmer = SpanishStemmer()

    with_hyphen = 0
    if "-" in word:
        with_hyphen = 1

    with_apostrophe = 0
    if "'" in word:
        with_apostrophe = 1

    o = str(o)
    features = [
        (o + "word", word),
        (o + 'pos', fields[1]),
        #(o + 'prefix1', word[:1]),
        (o + 'prefix2', word[:2]),
        (o + 'prefix3', word[:3]),
        (o + 'prefix4', word[:4]),
        #(o + 'suffix1', word[-1:]),
        (o + 'suffix2', word[-2:]),
        (o + 'suffix3', word[-3:]),
        (o + 'suffix4', word[-4:]),
        (o + 'is_upper', word.isupper()),
        (o + 'is_title', word.istitle()),
        (o + 'is_digit', word.isdigit()),
        (o + 'with_hypen', with_hyphen),
        (o + 'with_apostrophe', with_apostrophe),
        (o + 'spanich_stem', stemmer.stem(word)),
        # (o + 'word_shape', word_shape(word))
    ]

    return features


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-2, -1, 0, 1, 2]:
        if i + o >= 0 and i + o < len(sent):
            fields = sent[i + o]
            featlist = getfeats(fields, o)
            features.extend(featlist)

    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    train_sents = train_sents + dev_sents
    test_sents = list(conll2002.iob_sents("esp.testb"))

    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    model = SGDClassifier(loss = 'hinge', alpha = 0.00001, max_iter = 100)
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to sky-is-the-limit_results.txt")
    # format is: word gold pred
    with open("sky-is-the-limit_results.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py sky-is-the-limit_results.txt")
