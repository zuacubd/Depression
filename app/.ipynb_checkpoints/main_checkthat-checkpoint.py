# -*- coding: utf-8 -*-

import os
import pickle
import io
import collections
import argparse
import decimal


import numpy as np
import spacy
#from bert_embedding import BertEmbedding
from gensim import utils
from gensim.models import KeyedVectors
from gensim.models import word2vec
from sklearn import ensemble
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from ranking.rank_svm import RankSVM

from criterias.controversy import Controversy
from criterias.emotion import Emotion
from criterias.factuality_opinion import FactualityOpinion
from criterias.technicality import Technicality
from criterias.category import Category

#cwd = os.getcwd()
from path_manager import Datapath


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def parse(filename):
    f = open(filename, encoding="utf8")
    array = []

    for line in f:
        parts = line.split('\t')
        parts[0] = int(parts[0])
        parts[len(parts) - 1] = parts[len(parts) - 1].replace('\n', '')
        parts.append(0)
        array.append(parts)

    return array


def parse_labeled_file(filename):
    f = open(filename, encoding="utf8")
    array = []

    for line in f:
        parts = line.split('\t')
        parts[0] = int(parts[0])
        parts[len(parts) - 1] = parts[len(parts) - 1].replace('\n', '')
        parts[len(parts) - 1] = int(parts[len(parts) - 1])
        array.append(parts)

    return array


def load_fasttext_vectors(path):
    '''
        load glove trained token and its vector
    '''
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])), dtype=np.flaot32)
    return data


def load_glove_vectors(path):
    '''
        load glove trained token and its vector
    '''
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])), dtype=np.float32)
    return data


def calculer_score(text, controversy, factuality, technicality):
    controversy_score = controversy.score(text)
    #fact_score = FactualityOpinion(nlp).classify(text)
    fact_score = factuality.classify(text)
    technicality_score = technicality.score(text)
    emotion_pos_score, emotion_neg_score = Emotion.get_score(text)
    # print(text+" calculé")
    return [controversy_score, fact_score, technicality_score, emotion_pos_score, emotion_neg_score]


def same_speaker(speaker1, speaker2):
    if speaker1 == "SYSTEM" or speaker2 == "SYSTEM":
        return 2
    elif speaker1 != speaker2:
        return 1

    return 0


def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()


def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()

    return file


def normalize(predictions, score):
    max_ = max(predictions)
    min_ = min(predictions)

    max_ = max_ - min_
    if max_ == 0.:
        return score
    score = score - min_
    #score = score/(max_ - min_)
    score = score / max_
    return score


def divide_into_sentences(document):
    return [sent for sent in document.sents]


def number_of_fine_grained_pos_tags(sent):
    """
    Find all the tags related to words in a given sentence. Slightly more
    informative then part of speech tags, but overall similar data.
    Only one might be necessary.
    For complete explanation of each tag, visit: https://spacy.io/api/annotation
    """
    tag_dict = {
        '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0,
        '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
        'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0,
        'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0,
        'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0,
        'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0,
        'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
        'OOV': 0, 'TRAILING_SPACE': 0}

    for token in sent:
        if token.is_oov:
            tag_dict['OOV'] += 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE'] += 1
        else:
            tag_dict[token.tag_] += 1

    return tag_dict


def number_of_dependency_tags(sent):
    """
    Find a dependency tag for each token within a sentence and add their amount
    to a distionary, depending how many times that particular tag appears.
    """
    dep_dict = {
        'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
        'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
        'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
        'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
        'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
        'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
        'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
        'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
        'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0,
        'TRAILING_DEP': 0}

    for token in sent:
        if token.dep_ == '':
            dep_dict['TRAILING_DEP'] += 1
        else:
            try:
                dep_dict[token.dep_] += 1
            except:
                print('Unknown dependency for token: "' + token.orth_ + '". Passing.')

    return dep_dict


def number_of_specific_entities(sent):
    """
    Finds all the entities in the sentence and returns the amont of
    how many times each specific entity appear in the sentence.
    """
    entity_dict = {
        'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
        'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
        'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
        'ORDINAL': 0, 'CARDINAL': 0}

    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity] += 1

    return entity_dict


def number_of_proper_noun_entities(sent):
    """
    Finds all the proper noun entities in the sentence and returns the amount of
    how many times each specific entity appear in the sentence.
    """
    pn_entity_dict = {
        'PER': 0, 'ORG': 0, 'LOC': 0, 'MISC': 0}

    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        pn_entity_dict[entity] += 1

    return pn_entity_dict


def get_df(test_sent):
    # Preprocess using spacy
    parsed_test = divide_into_sentences(nlp(test_sent))
    if len(parsed_test) < 0:
        parsed_test.append('')

    #Get features
    sentence_with_features = {}

    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)

    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)

    dep_dict = number_of_dependency_tags(parsed_test[0])
    sentence_with_features.update(dep_dict)

    df = np.fromiter(iter(sentence_with_features.values()), dtype=float)

    return df.reshape(1, -1)


def get_wiki_df(test_sent):
    # Preprocess using spacy
    parsed_test = divide_into_sentences(nlp_wiki(test_sent))
    if len(parsed_test) < 0:
        parsed_test.append('')
    # Get features
    sentence_with_features = {}

    pn_entities_dict = number_of_proper_noun_entities(parsed_test[0])
    sentence_with_features.update(pn_entities_dict)

    wiki_df = np.fromiter(iter(sentence_with_features.values()), dtype=float)

    return wiki_df.reshape(1, -1)


def get_cf(test_sent):

    sentence_with_features = {}

    categories_dict = category.get_categories(test_sent)
    sentence_with_features.update(categories_dict)

    cf = np.fromiter(iter(sentence_with_features.values()), dtype=float)

    return cf.reshape(1, -1)


def sentence_features(sentence):
    features = []

    if use_label:
        features = [calculer_score(sentence, controversy, factuality, technicality)]

    if use_wiki_spacy:
        features = np.append(features, get_wiki_df(sentence)[0])

    if use_spacy:
        features = np.append(features, get_df(sentence)[0])

    if use_category:
        features = np.append(features, get_cf(sentence)[0])

    if use_fact_w2v:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(word_vectors_fact.wv[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(word_vectors_fact.wv["tax"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_w2v_wiki:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(word_vectors_wiki.wv[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(word_vectors_wiki.wv["the"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_w2v:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(word_vectors.wv[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(word_vectors.wv["tax"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_glove:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(glove_vectors[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(glove_vectors["the"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_glove_cc:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(glove_vectors_cc[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(glove_vectors_cc["the"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_fasttext:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(fasttext_vectors[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(fasttext_vectors["the"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))


    if use_fasttext_cc:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(fasttext_vectors_cc[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(fasttext_vectors_cc["the"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    if use_bert:
        text = sentence.split('\n')
        sentence_vector = []
        for line in text:
            try:
                model = bert_vectors(line)
                result = model[0]
                token = result[0]
                vectors = result[1]
                sentence_vector.append(np.mean(vectors, axis=0))
            except:
                pass

        if len(sentence_vector)>1:
            features = np.append(features, np.mean(sentence_vector, axis=0))
        else:
            features = np.append(features, sentence_vector[0])

    return features


def trainSet(train_data):
    X = []
    y = []
    vectors = []

    if speakers:
        speakers_arr = []
        for i in train_data:
            speakers_arr.append(i[1])
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(train_data)):
            x_i = []

            for previous in range(surround_scope, 0, -1):
                if int(train_data[i][0]) - previous > 0:
                    x_i = np.append(np.append(x_i, vectors[i - previous]),
                                    [same_speaker(speakers_arr[i], speakers_arr[i - previous])])
                else:
                    x_i = np.append(np.append(x_i, np.zeros_like(vectors[i])), [2])

            x_i = np.append(np.append(x_i, vectors[i]), [0])

            for next in range(surround_scope):
                if i + next + 1 >= len(train_data) or train_data[i][0] + next + 1 > int(
                        train_data[i + 1][
                            0]):
                    x_i = np.append(np.append(x_i, np.zeros_like(vectors[i])), [2])
                else:
                    x_i = np.append(np.append(x_i, vectors[i + next + 1]),
                                    [same_speaker(speakers_arr[i], speakers_arr[i + next + 1])])

            X.append(x_i)
            y.append(train_data[i][3])
    else:
        for i in train_data:
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(train_data)):
            x_i = []

            for previous in range(surround_scope, 0, -1):
                if int(train_data[i][0]) - previous > 0:
                    x_i = np.append(x_i, vectors[i - previous])
                else:
                    x_i = np.append(x_i, np.zeros_like(vectors[i]))

            x_i = np.append(x_i, vectors[i])

            for next in range(surround_scope):
                if i + next + 1 >= len(train_data) or train_data[i][0] + next + 1 > int(
                        train_data[i + 1][
                            0]):
                    x_i = np.append(x_i, np.zeros_like(vectors[i]))
                else:
                    x_i = np.append(x_i, vectors[i + next + 1])

            X.append(x_i)
            y.append(train_data[i][3])

    return X, y

def testSet(data):
    to_predict = []
    vectors = []

    if speakers:
        speakers_arr = []
        for i in data:
            speakers_arr.append(i[1])
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(data)):
            to_predict.append([])

            for previous in range(surround_scope, 0, -1):
                if i - previous >= 0:
                    to_predict[i] = np.append(np.append(to_predict[i], vectors[i - previous]),
                                              [same_speaker(speakers_arr[i], speakers_arr[i - previous])])
                else:
                    to_predict[i] = np.append(np.append(to_predict[i], np.zeros_like(vectors[i])), [2])

            to_predict[i] = np.append(np.append(to_predict[i], vectors[i]), [0])

            for next in range(surround_scope):
                if i + next + 1 < len(data):
                    to_predict[i] = np.append(np.append(to_predict[i], vectors[i + next + 1]),
                                              [same_speaker(speakers_arr[i], speakers_arr[i + next + 1])])
                else:
                    to_predict[i] = np.append(np.append(to_predict[i], np.zeros_like(vectors[i])), [2])
    else:
        for i in data:
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        to_predict = []
        for i in range(len(data)):
            to_predict.append([])

            for previous in range(surround_scope, 0, -1):
                if i - previous >= 0:
                    to_predict[i] = np.append(to_predict[i], vectors[i - previous])
                else:
                    to_predict[i] = np.append(to_predict[i], np.zeros_like(vectors[i]))

            to_predict[i] = np.append(to_predict[i], vectors[i])

            for next in range(surround_scope):
                if i + next + 1 < len(data):
                    to_predict[i] = np.append(to_predict[i], vectors[i + next + 1])
                else:
                    to_predict[i] = np.append(to_predict[i], np.zeros_like(vectors[i]))

    return to_predict

def predictAndStore(data, to_predict, classifier, output_file):
    '''Classifying the given data and store the results'''
    predictions = []
    for i in range(len(data)):
        prediction = classifier.predict_proba([to_predict[i]])
        if len(prediction[0])==1:
            predictions.append(prediction[0][0])
        else:
            predictions.append(prediction[0][1])

    for i in range(len(data)):
        sentence_id = data[i][0]
        score = predictions[i]
        #score = normalize(predictions, predictions[i])
        output_file.write(str(sentence_id) + "\t" + float_to_str(score) + "\n")
    output_file.close()


def predictTrainSet(train_files, Features):
    num_sample = []
    X = []
    start = 0
    end = 0

    for fdx in range(0, len(train_files)):
        filename = train_files[fdx]
        print ("File: {}".format(filename))
        data = parse_labeled_file(os.path.join(data_raw_dir, "task1_train/" + filename))
        num_sample.append(len(data))

        end = start + num_sample[fdx]
        X.append(Features[start:end])
        start = end

    if not os.path.exists(os.path.join(output_dir, 'train/' + str(feature_acro))):
        os.makedirs(os.path.join(output_dir, 'train/' + str(feature_acro)))

    approaches = ['natural', 'oversampled', 'undersampled', 'combined_ou', 'balanced']
    for fdx in range(0, len(train_files)):
        filename = train_files[fdx]
        data = parse_labeled_file(os.path.join(data_raw_dir, "task1_train/" + filename))
        to_predict = X[fdx]

        for model in All_models:

            if len(model) > 2:
                method = model[0]
                natural = False
                oversampled = model[1]
                undersampled = model[2]
                combined_ou = model[3]
                balanced = model[4]
                approaches_trained = [natural, oversampled, undersampled, combined_ou, balanced]
                for adx in range(0, len(approaches_trained)):
                    if approaches_trained[adx]:
                        approach = method + '_' + approaches[adx]
                        classifier = load_pickle(os.path.join(models_dir, feature_acro + '/' + approach  + "_classifier.pickle"))
                        output_file = open(os.path.join(output_dir, "train/" + str(feature_acro) + '/' + filename[:-4] + "_" +  approach + ".txt"), 'w')
                        predictAndStore(data, to_predict, classifier, output_file)
            else:
                approach = model[0] + '_' + approaches[0]
                classifier = load_pickle(os.path.join(models_dir, feature_acro + '/' + approach + "_classifier.pickle"))
                output_file = open(os.path.join(output_dir, "train/" + str(feature_acro) + '/' + filename[:-4] + "_" +  approach + ".txt"), 'w')
                predictAndStore(data, to_predict, classifier, output_file)


def predictTestSet(test_files):
    '''Predict the test data on the trained models'''
    #handle feature selection case

    approaches = ['natural', 'oversampled', 'undersampled', 'combined_ou', 'balanced']
    for fdx in range(0, len(test_files)):
        filename = test_files[fdx]
        data = parse(os.path.join(data_raw_dir, "task1_test/english/" + filename))

        if not os.path.exists(os.path.join(features_dir, 'test/' + str(feature_acro) + filename)):
            to_predict = testSet(data)
            write_list(to_predict, os.path.join(features_dir, 'test/' + str(feature_acro) + filename))
        else:
            to_predict = loadTestSet(os.path.join(features_dir, 'test/' + str(feature_acro) + filename))

        if feature_selection:
            X_predict = np.asmatrix(to_predict)
            X_selected = X_predict[:,selected_features_index]
            to_predict = X_selected.tolist()

        if not os.path.exists(os.path.join(output_dir, 'test/' + str(feature_acro))):
            os.makedirs(os.path.join(output_dir, 'test/' + str(feature_acro)))

        for model in All_models:
            if len(model) > 2:
                method = model[0]
                natural = False
                oversampled = model[1]
                undersampled = model[2]
                combined_ou = model[3]
                balanced = model[4]
                approaches_trained = [natural, oversampled, undersampled, combined_ou, balanced]
                for adx in range(0, len(approaches_trained)):
                    if approaches_trained[adx]:
                        approach = method + '_' + approaches[adx]
                        classifier = load_pickle(os.path.join(models_dir, feature_acro + '/' + approach  +
                                                              "_classifier.pickle"))
                        output_file = open(os.path.join(output_dir, 'test/' + feature_acro + '/' + filename[:-4] + "_" +
                                                        approach + ".txt"), 'w')
                        predictAndStore(data, to_predict, classifier, output_file)
            else:
                approach = model[0] + '_' + approaches[0]
                classifier = load_pickle(os.path.join(models_dir, feature_acro + '/' +
                                         approach + "_classifier.pickle"))
                output_file = open(os.path.join(output_dir, 'test/' + feature_acro + '/' +
                                   filename[:-4] + "_" + approach + ".txt"), 'w')
                predictAndStore(data, to_predict, classifier, output_file)


def loadTrainSet(path_features, path_labels):
    X = load_pickle(path_features)
    y = load_pickle(path_labels)
    return X, y


def loadTestSet(path_features):
    X = load_pickle(path_features)
    return X

def write_list(list_data, file_name):
    '''
        writing a list to a file where each item of the list is stored in a single line.
        '''
    with open(file_name, 'wb') as fh:
        #for item in list_data:
        #    fh.write("{}\n".format(item[0]))
        pickle.dump(list_data, fh)

def get_features_std(names, X):
    r, c = X.shape
    stds = []
    for j in range(0, c):
        feature_vector = X[:,j]
        val = np.std(feature_vector)
        stds.append(val)
    features_std = dict(zip(names, stds))
    return features_std

def get_feature_selection(X, selected_features_path):
    '''Feature selection strategy'''
    X_mat = np.asmatrix(X)
    row, col = X_mat.shape
    '''
    names = [i for i in range(col)]
    features_std = get_features_std(names, X_mat)
    ordered_features_std = collections.OrderedDict(sorted(features_std.items(), key=lambda t: t[1], reverse=True))

    selected_features_index = []
    for i, (key,val) in enumerate(ordered_features_std.items()):
        if val > 0.0:
            selected_features_index.append(key)
    print (selected_features_index)
    top_selected_features_index = selected_features_index[:300]
    '''
    with open(selected_features_path, 'r') as fr:
        lines = fr.readlines()
    top_selected_features_index = [int(line) for line in lines]
    X_selected = X_mat[:,top_selected_features_index]
    return top_selected_features_index, X_selected.tolist()

def trainModel(X, y, model, feature_acro):

    if len(model) == 2:
        method = model[0]
        natural = model[1]
        oversampled = False
        undersampled = False
        combined_ou = False
        balanced = False

    else:
        method = model[0]
        natural = False
        oversampled = model[1]
        undersampled = model[2]
        combined_ou = model[3]
        balanced = model[4]

    kindSMOTE = 'regular'

    if method == 'rank_svm':
        classifier = RankSVM()
    elif method == 'random_forest':
        classifier = ensemble.RandomForestClassifier(random_state=42)
    elif method == 'svc_rbf':
        classifier = svm.SVC(probability=True, random_state=42)
        kindSMOTE = 'svm'
    elif method == 'knn3':
        classifier = neighbors.KNeighborsClassifier(3, weights = 'uniform')
    elif method =='log_reg':
        classifier = LogisticRegression(random_state=42, class_weight='balanced')
        kindSMOTE = 'svm'
    elif method == 'sgd_log':
        classifier = SGDClassifier(loss='log', random_state=42)
        kindSMOTE = 'svm'
    elif method == 'nn_lbfgs':
        classifier = MLPClassifier(solver='lbfgs', random_state=42)
    else:
        classifier = svm.SVC(probability=True, kernel='linear', random_state=42)
        kindSMOTE = 'svm'

    if not os.path.exists(os.path.join(models_dir, feature_acro)):
        os.makedirs(os.path.join(models_dir, feature_acro))

    if oversampled:
        approach = method + '_oversampled'
        # print("Training " + method)
        from imblearn.over_sampling import SVMSMOTE
        smote = SVMSMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
        save_pickle(classifier, os.path.join(models_dir, feature_acro +'/' + approach + '_classifier.pickle'))

    if undersampled:
        approach = method + '_undersampled'
        # print("Training " + method)
        from imblearn.under_sampling import EditedNearestNeighbours
        enn = EditedNearestNeighbours()
        X_resampled, y_resampled = enn.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
        save_pickle(classifier, os.path.join(models_dir, feature_acro +'/' + approach + '_classifier.pickle'))

    if combined_ou:
        approach = method + '_combined_ou'
        # print("Training " + method)
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smt.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
        save_pickle(classifier, os.path.join(models_dir, feature_acro +'/' + approach + '_classifier.pickle'))

    if balanced:
        approach = method + '_balanced'
        # print("Training " + method)
        from imblearn.ensemble import BalancedRandomForestClassifier
        brf = BalancedRandomForestClassifier(max_depth=2, random_state=42)
        #X_resampled, y_resampled = smt.fit_sample(X, y)
        brf.fit(X, y)
        save_pickle(brf, os.path.join(models_dir, feature_acro +'/' + approach + '_classifier.pickle'))

    if natural:
        # print("Training " + method)
        approach = method + '_natural'
        classifier.fit(X, y)
        save_pickle(classifier, os.path.join(models_dir, feature_acro +'/' + approach + '_classifier.pickle'))


def get_process_flag():
    '''
        returns the flag for the train/test/predict process
    '''
    train = False
    test = False
    predict = False

    if process == 1:
        train = True

    elif process == 2:
        test = True

    elif process == 3:
        predict = True

    else:
        print ('Incorrect process number! Please choose 1 (train/learning), 2 (testing) or 3 (prediction)')

    return train, test, predict


def get_feature_acronym():
    '''
        returns the acronym for features
    '''

    feature_acro = ''
    if use_label:
        feature_acro = feature_acro + 'N'
    if use_spacy:
        feature_acro = feature_acro + 'L'
    if use_wiki_spacy:
        feature_acro = feature_acro + 'E'
    if use_category:
        feature_acro = feature_acro + 'C'
    if use_fact_w2v:
        feature_acro = feature_acro + 'F'
    if use_w2v:
        feature_acro = feature_acro + 'W'
    if use_w2v_wiki:
        feature_acro = feature_acro + 'Ww'
    if use_glove:
        feature_acro = feature_acro + 'G'
    if use_glove_cc:
        feature_acro = feature_acro + 'Gc'
    if use_fasttext:
        feature_acro = feature_acro + 'Ft'
    if use_fasttext_cc:
        feature_acro = feature_acro + 'Ftc'
    if use_bert:
        feature_acro = feature_acro + 'B'

    return feature_acro


def argument_parser():
    '''
        returns the parser to parse the arguments received by the program
    '''
    parser = argparse.ArgumentParser(description='A tool used to compute information check-worthiness ...')

    parser.add_argument('-data_dir', '--data_folder_path', nargs='?', type=str, required=True,
                        help='The folder path which contains different raw data, processed data, models, results, etc.')
    parser.add_argument('-p', '--process', nargs='?', type=int, required=True,
                        help='train/learning (1), testing (2) or prediction (3) process to be performed')
    return parser


if __name__ == '__main__':

    print ("processing arguments ...")
    parser = argument_parser()
    # Parsing the args
    args = parser.parse_args()

    # Retrieving the args
    data_folder_path = args.data_folder_path
    print("Data folder: {}".format(data_folder_path))

    process = args.process
    print("Process: {}".format(process))

    dataPath = Datapath(data_folder_path)
    root_dir = dataPath.get_root_dir()
    data_dir = dataPath.get_data_dir()
    data_raw_dir = dataPath.get_data_raw_dir()
    resources_dir = dataPath.get_resources_dir()
    features_dir = dataPath.get_features_dir()
    models_dir = dataPath.get_models_dir()
    results_dir = dataPath.get_results_dir()
    output_dir = dataPath.get_output_dir()

    # Max number of digits for the computed scores
    ctx = decimal.Context()
    ctx.prec = 20

    # Number of sentences before and after the one to evaluate that we take in account
    surround_scope = 0
    # True if we take in account the name of the speaker
    speakers = False

    # Methods used
    use_label = True
    use_spacy = False
    use_category = False
    use_wiki_spacy = False
    use_fact_w2v = False
    use_w2v = False
    use_w2v_wiki = False
    use_glove = False
    use_glove_cc = False
    use_fasttext = False
    use_fasttext_cc = False
    use_bert = False

    feature_acro = get_feature_acronym()
    train, test, predict = get_process_flag()

    ########################################### feature selection ##############
    feature_selection = False #fs

    # True if we want to average the scores given by each model
    combine = False
    train_files = ['Task1-English-1st-Presidential.txt','Task1-English-2nd-Presidential.txt',
                   'Task1-English-Vice-Presidential.txt']

    test_files = ["task1-en-file1.txt","task1-en-file2.txt","task1-en-file3.txt","task1-en-file4.txt",
                  "task1-en-file5.txt","task1-en-file6.txt","task1-en-file7.txt",]

    All_models = [['random_forest', False, False, False, True], ['random_forest', True], ['svc_rbf', True], ['knn3', True], ['log_reg', True], ['sgd_log', True], ['nn_lbfgs', True], ['svc_linear', True]]
    #All_models = [['random_forest', False, False, False, True]]

    if train or test:
        if use_label:
            controversy = Controversy(resources_dir)
            technicality = Technicality(resources_dir)

        if use_label or use_spacy:
            model_size = 'md'
            nlp = spacy.load('en_core_web_' + model_size)
            factuality = FactualityOpinion(nlp, models_dir)
            # print(model_size + " model loaded")

        if use_wiki_spacy:
            nlp_wiki = spacy.load("xx_ent_wiki_sm", disable=["parser"])
            nlp_wiki.add_pipe(nlp_wiki.create_pipe('sentencizer'))

        if use_category:
            category = Category()

        if use_fact_w2v:
            word_vectors_fact = word2vec.Word2Vec.load(os.path.join(root_dir,
                                                                    "word_embedding/facts/word2vec_model/model1"))

        if use_w2v:
            word_vectors = KeyedVectors.load_word2vec_format(os.path.join(root_dir,
                        "word_embedding/pretrain/word2vec/googlenews/GoogleNews-vectors-negative300.bin"), binary=True)

        if use_w2v_wiki:
            path = os.path.join(root_dir, "word_embedding/pretrain/word2vec/wiki/wiki.en.model_w5")
            word_vectors_wiki = KeyedVectors.load(path)

        if use_glove:
            path = os.path.join(root_dir, "word_embedding/pretrain/glove/wiki_gigaword/glove.6B.300d.txt")
            # print
            glove_vectors = load_glove_vectors(path)
            # print("glove loaded")

        if use_glove_cc:
            path = os.path.join(root_dir, "word_embedding/pretrain/glove/common_crawl42B/glove.42B.300d.txt")
            # print
            glove_vectors_cc = load_glove_vectors(path)
            # print("glove loaded")

        if use_fasttext:
            path = os.path.join(root_dir, "word_embedding/pretrain/fasttext/wiki-news/wiki-news-300d-1M.vec")
            fasttext_vectors = load_glove_vectors(path)
            # print("fasttext loaded")

        if use_fasttext_cc:
            path = os.path.join(root_dir, "word_embedding/pretrain/fasttext/common_crawl/crawl-300d-2M.vec")
            fasttext_vectors_cc = load_glove_vectors(path)
            # print("fasttext loaded")

        if use_bert:
            #path = cwd + "/word_embedding/pretrain/fasttext/wiki-news/wiki-news-300d-1M.vec"
            bert_vectors = BertEmbedding()
            # print("bert loaded") 

    # The train data is in "train_data.txt"
    train_data_array = parse_labeled_file(os.path.join(data_raw_dir, "task1_train/train_data.txt"))

    if not os.path.exists(os.path.join(features_dir, 'train/' + str(feature_acro))):
        X, y = trainSet(train_data_array)
        #store train set features
        write_list(X, os.path.join(features_dir, 'train/' + str(feature_acro)))
        write_list(y, os.path.join(features_dir, 'train/' + str(feature_acro) + '_labels'))

    else:
        X, y = loadTrainSet(os.path.join(features_dir, 'train/' + str(feature_acro)),
                            os.path.join(features_dir, 'train/' + str(feature_acro) + '_labels'))

    if feature_selection:
        if use_spacy or use_category:
           feature_acro = feature_acro + 'fs_svml1_f1'
        selected_features_path = os.path.join(features_dir, 'fs/train-features-svm-l1-norm-f1-C10.txt')
        selected_features_index, X_selected = get_feature_selection(X, selected_features_path)

    #training models
    if train:
        for model in All_models:
            if feature_selection:
                trainModel(X_selected, y, model, feature_acro)
            else:
                trainModel(X, y, model, feature_acro)

    #prediction for the train set
    if train:
        if feature_selection:
            predictTrainSet(train_files, X_selected)
        else:
            predictTrainSet(train_files, X)

    #prediction for the test set
    if test:
        predictTestSet(test_files)
