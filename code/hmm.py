from nltk.corpus import conll2002
from collections import  defaultdict
import copy
from hmmlearn import hmm
import numpy as np



states= ['I-ORG', 'B-ORG', 'B-LOC', 'O', 'I-MISC', 'B-PER', 'B-MISC', 'I-PER', 'I-LOC']
word_ls = []

def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + "word", word)
        # TODO: add more features here.
    ]
    return features

def transition_matrix(transitions):
    n = len(states) #number of states
    M = [[0]*n for _ in range(n)]
    not_last_classes = [0 for i in range(9)]

    for (i,j) in zip(transitions,transitions[1:]):
        i_idx = states.index(i)
        j_idx = states.index(j)
        M[i_idx][j_idx] += 1
        not_last_classes[i_idx] += 1
    # print(dict_mat)

    for j, row in enumerate(M):
        for i in range(9):
            row[i] /= not_last_classes[j]

    # for j, row in M:
    #     s = sum(row)
    #     if s > 0:
    #         row[:] = [f/s for f in row]
    return M

# get initial states' probabilities
def get_initial_prob(init_states):
    states_dict = {st: 0 for st in states}
    for st in init_states:
        states_dict[st] += 1

    total_sent = len(init_states)
    init_probs = []
    for st in states:
        prob = states_dict[st]/total_sent
        init_probs.append(prob)

    return init_probs


def get_emission_prob(observations, wrd_ls):
    M = [[0] * len(wrd_ls) for _ in range(len(states))]

    st_dict = {st: None for st in states}
    for (w, s) in observations:
        s_idx = states.index(s)
        w_idx = wrd_ls.index(w)
        M[s_idx][w_idx] += 1


    #calculate emisssion probabilities
    emission_mat = copy.deepcopy(M)
    for s in range(len(states)):
        total = sum(M[s])
        for w in range(len(wrd_ls)):
            prob = M[s][w]/total
            emission_mat[s][w] = prob

    return emission_mat

def get_feature_mat(sents):
    sent_features = []
    length = []
    for s in sents:
        len_s = len(s)
        for i in range(len_s):
            w = s[i][0]
            boi = s[i][-1]
            if w not in word_ls:
                # sent_features.append((word_ls.index("UNK"), states.index(boi)))
                sent_features.append(word_ls.index("UNK"))
            else:
                # sent_features.append((word_ls.index(w), states.index(boi)))
                sent_features.append(word_ls.index(w))
        length.append(len_s)

    # print(len(length))

    return sent_features, np.array(length)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    train_sents = train_sents + dev_sents
    test_sents = list(conll2002.iob_sents("esp.testb"))


    # get a list of states
    states_ls = []
    sent_inits_states = []
    observations = []
    word_set = set()
    for sent in train_sents:
        init_state = sent[0][-1]
        sent_inits_states.append(init_state)
        # sent = []
        for i in range(len(sent)):
            word = sent[i][0]
            word_set.add(word)
            boi = sent[i][-1]
            observations.append((word, boi))
            # sent.append((word, boi))
            states_ls.append(boi)

    word_ls = list(word_set)
    word_ls.append("UNK")
    init_probs = get_initial_prob(sent_inits_states)
    emission_mat = get_emission_prob(observations, word_ls)
    m = transition_matrix(states_ls)
    # for row in m:
    #     print(' '.join('{0:.5f}'.format(x) for x in row))


    init_probs = np.array(init_probs).astype(np.float64)
    # print('start probability: ' ,  init_probs.shape)
    transition_mat = np.array(m).astype(np.float64)
    # print('transition matrix: ' , transition_mat.shape)
    emission_mat = np.array(emission_mat).astype(np.float64)
    # print('emission probability: ' , emission_mat.shape)


    # X = np.array([(word_ls.index(x), states.index(y))for x, y in observations])
    X_train, len_train = get_feature_mat(train_sents)
    X_test, len_test = get_feature_mat(test_sents)


    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)

    # print(X_train)
    # print(X_train.shape)
    model = hmm.MultinomialHMM(n_components = len(states), algorithm = 'viterbi')
    model.startprob_ = init_probs
    model.transmat_ = transition_mat
    model.emissionprob_ = emission_mat
    _, y_pred = model.decode(X_test, lengths = len_test)

    j = 0
    print("Writing to results_hmm.txt")
    # format is: word gold pred
    with open("results_hmm.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = states[y_pred[j]]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results_hmm.txt")
