import numpy as np
from deslib.util.diversity import double_fault
from sklearn.manifold import TSNE
from deslib.util.aggregation import *
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import matplotlib.lines as mlines
# Tensorflow


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def calc_delta(labels={}, pred={}, clfs=[]):
    size = len(clfs)
    delta = np.zeros(shape=(size, size))
    for i, clf1 in enumerate(clfs):
        for k, clf2 in enumerate(clfs):
            delta[i][k] = double_fault(labels, pred[clf1], pred[clf2])
    return delta, clfs


def gera_tsne(delta=[], title='', classifiers=[], size=5, escala=0, x_ini=0, y_ini=0, x_fim=0, y_fim=0, legend=False):
    simbolos = {}
    simbolos[0] = 'X'
    simbolos[1] = 'd'
    simbolos[2] = '*'
    simbolos[3] = "^"
    simbolos[4] = 'o'

    tsne_model = TSNE(perplexity=50, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(delta)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(size, size))
    plt.title(title)
    if escala > 0:
        plt.ylim(escala * -1, escala)
        plt.xlim(escala * -1, escala)
    else:
        plt.ylim(y_ini, y_fim)
        plt.xlim(x_ini, x_fim)

    dot = 0
    for i in range(len(x)):
        plt.scatter(x[i], y[i], marker=simbolos[dot], label=classifiers[i], s=100)
        dot = dot + 1
    if legend:
        plt.legend()
    plt.show()


def get_classifier(clf, statement, label, ext):
    pipe_clf = Pipeline([
        ('extractor', ext),
        ('clf', clf)
    ])
    pipe_clf.fit(statement, label)
    return pipe_clf
# CNN


def get_CNN(ext, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=300, MAX_SEQUENCE_LENGTH=300, activation='sigmoid', word_embedding=False, dense=2):
    if word_embedding == False:
        X_ext = ext.get_feature_names()
        model = Word2Vec([X_ext], min_count=1, workers=1, size=300)
    else:
        model = ext.model

    word_index = tokenizer.word_index

    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in model.wv.vocab:
            embedding_matrix[i] = model.wv.word_vec(word)

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False
                                )
    cnn = Sequential()
    cnn.add(embedding_layer)
    cnn.add(Dropout(0.2))
    cnn.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    cnn.add(GlobalMaxPooling1D())
    cnn.add(Dense(256))
    cnn.add(Dropout(0.2))
    cnn.add(Activation('relu'))
    cnn.add(Dense(dense))
    cnn.add(Activation(activation))
    cnn.summary()
    cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return cnn
#


def get_oracle(labels, pred={}, clfs=[]):
    cont = 0
    list_yes = np.zeros((len(labels)))
    for i, k in enumerate(labels):
        acertou = False
        for clf in clfs:
            if pred[clf][i] == k:
                acertou = True
                break
        if acertou:
            cont += 1
            list_yes[i] = k
        else:
            if k == 0:
                list_yes[i] = 1
            else:
                list_yes[i] = 0
    return cont, list_yes


def tsne_full(labels,  delta, title='', escala=0, size=5, x_ini=0, x_fim=0, y_ini=0, y_fim=0, espacamento=50, per=100, lern=200.0, iterations=2500, ang=0.5, size_dot=10, cores={}, simbolos={}):
    tsne_model = TSNE(init='pca', early_exaggeration=espacamento, perplexity=per, learning_rate=lern, random_state=42, n_iter=iterations, angle=ang)
    new_values = tsne_model.fit_transform(delta)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(size, size))
    plt.title(title)

    if escala > 0:
        plt.ylim(escala * -1, escala)
        plt.xlim(escala * -1, escala)
    else:
        plt.ylim(y_ini, y_fim)
        plt.xlim(x_ini, x_fim)

    dot = 0
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=size_dot,  marker=simbolos[dot], c=cores[dot])

        if labels[i] == 'SVM-CV' and i == 40:
            posicao = (5, -20)
        elif labels[i] == 'MNB-GL':
            posicao = (5, 25)
        elif i == 56:
            posicao = (5, 25)
        else:
            posicao = (5, 8)

        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=posicao, textcoords='offset points', ha='center', va='bottom')
        dot = dot + 1
        if dot == 5:
            dot = 0

    m1 = mlines.Line2D([], [], color=cores[0], marker=simbolos[0], linestyle='None', markersize=10, label='CV')
    m2 = mlines.Line2D([], [], color=cores[1], marker=simbolos[1], linestyle='None', markersize=10, label='TFIDF')
    m3 = mlines.Line2D([], [], color=cores[2], marker=simbolos[2], linestyle='None', markersize=10, label='Word2Vec')
    m4 = mlines.Line2D([], [], color=cores[3], marker=simbolos[3], linestyle='None', markersize=10, label='Glove')
    m5 = mlines.Line2D([], [], color=cores[4], marker=simbolos[4], linestyle='None', markersize=10, label='FastText')

    plt.legend(handles=[m1, m2, m3, m4, m5])
    plt.show()


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(
            rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom'
        )


def render_bar(list_certos, title='', size_x=0, size_y=0):
    labels = [l for l in range(len(list_certos) - 1, -1, -1)]

    # labels = [5, 4, 3, 2, 1, 0]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, list_certos.values(), width)

    ax.set_ylabel('Instâncias')
    ax.set_xlabel('Classificadores')
    ax.set_title('Classificadores x Instâncias: ' + title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    fig.tight_layout()
    if size_x != 0:
        plt.rcParams["figure.figsize"] = (size_y, size_x)
    plt.show()


def get_combine(data_test, clfs={}, cnn=False, test=None):
    list_proba = []

    for ext, clf in clfs.items():
        list_proba.append(clf.predict_proba(data_test))

    ensemble = np.array(list_proba).transpose((1, 0, 2))

    votes = np.zeros((test.shape[0], len(clfs)))
    k = 0
    for clf_index, clf in clfs.items():
        if cnn == False:
            votes[:, k] = clf.predict(data_test).reshape(test.shape[0])
        else:
            votes[:, k] = np.argmax(clf.predict(data_test), axis=1).reshape(test.shape[0])
        k += 1

    return majority_voting_rule(votes), average_rule(ensemble), maximum_rule(ensemble), minimum_rule(ensemble), median_rule(ensemble), product_rule(ensemble)


def get_combine_all(d_test, d_test_cnn, clfs={}):
    list_proba = []

    for key, clf in clfs.items():
        if key[:3] == 'CNN':
            list_proba.append(clf.predict_proba(d_test_cnn))
        else:
            list_proba.append(clf.predict_proba(d_test))

    ensemble = np.array(list_proba).transpose((1, 0, 2))

    votes = np.zeros((d_test.shape[0], len(clfs)))
    k = 0
    for key, clf in clfs.items():
        if key[:3] == 'CNN':
            votes[:, k] = np.argmax(clf.predict(d_test_cnn), axis=1).reshape(d_test.shape[0])
        else:
            votes[:, k] = clf.predict(d_test).reshape(d_test.shape[0])
        k += 1

    return majority_voting_rule(votes), average_rule(ensemble), maximum_rule(ensemble), minimum_rule(ensemble), median_rule(ensemble), product_rule(ensemble)
