from zeugma.embeddings import EmbeddingTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *
from function import *

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from deslib.util.aggregation import *
import pandas as pd


def load_embedding():
    w2v = EmbeddingTransformer('word2vec')
    glove = EmbeddingTransformer('glove')
    fasttext = EmbeddingTransformer('fasttext')

    return w2v, glove, fasttext


def get_svm(cv, tfidf, w2v, glove, fasttext):
    svm = {
        'CV': {
            'CLF': SVC(random_state=42, kernel='linear', gamma=0.1, probability=True),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': SVC(random_state=42, kernel='linear', gamma=0.1, probability=True),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': SVC(random_state=42, kernel='rbf', gamma=1, probability=True),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': SVC(random_state=42, kernel='rbf', gamma=0.5, probability=True),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': SVC(random_state=42, kernel='rbf', gamma=1, probability=True),
            'EXT': fasttext,
        }
    }
    return svm


def get_lr(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': LogisticRegression(random_state=42,
                                      multi_class='auto', solver='liblinear', penalty='l1'),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': LogisticRegression(random_state=42, multi_class='auto', solver='liblinear', penalty='l1'),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': LogisticRegression(random_state=42, multi_class='auto', solver='liblinear', penalty='l1'),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': LogisticRegression(random_state=42, multi_class='auto', solver='liblinear', penalty='l1'),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': LogisticRegression(random_state=42, multi_class='auto', solver='liblinear', penalty='l1'),
            'EXT': fasttext,
        }
    }


def get_rf(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': RandomForestClassifier(random_state=42, verbose=100, n_estimators=20, n_jobs=-1),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': RandomForestClassifier(random_state=42, verbose=100, n_estimators=50, n_jobs=-1),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': RandomForestClassifier(random_state=42, verbose=100, n_estimators=50, n_jobs=-1),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': RandomForestClassifier(random_state=42, verbose=100, n_estimators=50, n_jobs=-1),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': RandomForestClassifier(random_state=42, verbose=100, n_estimators=50, n_jobs=-1),
            'EXT': fasttext,
        }
    }


def get_nb(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': MultinomialNB(alpha=1, fit_prior=False),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': MultinomialNB(alpha=0.5, fit_prior=False),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': BernoulliNB(alpha=0.5, fit_prior=True),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': BernoulliNB(alpha=0.1, fit_prior=True),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': BernoulliNB(alpha=1, fit_prior=True),
            'EXT': fasttext,
        }
    }


def get_mlp(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': MLPClassifier(random_state=42, batch_size=20, max_iter=40,
                                 activation='relu', solver='lbfgs'),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': MLPClassifier(random_state=42, batch_size=20, max_iter=40,
                                 activation='logistic', solver='adam'),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': MLPClassifier(random_state=42, batch_size=20, max_iter=40,
                                 activation='relu', solver='adam'),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': MLPClassifier(random_state=42, batch_size=20, max_iter=40, activation='relu', solver='adam'),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': MLPClassifier(random_state=42, batch_size=20, max_iter=40, activation='relu', solver='adam'),
            'EXT': fasttext,
        }
    }


def get_extra(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'EXT': fasttext,
        }
    }


def get_knn(cv, tfidf, w2v, glove, fasttext):
    return {
        'CV': {
            'CLF': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            'EXT': cv,
        },
        'TFIDF': {
            'CLF': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'EXT': tfidf,
        },
        'W2V': {
            'CLF': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'EXT': w2v,
        },
        'GLOVE': {
            'CLF': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'EXT':  glove,
        },
        'FAST': {
            'CLF': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'EXT': fasttext,
        }
    }


def update_pred_proba(clfs, label, train, val, test):
    pred_train = pd.DataFrame()
    pred_val = pd.DataFrame()
    pred_test = pd.DataFrame()

    prob_train = pd.DataFrame()
    prob_val = pd.DataFrame()
    prob_test = pd.DataFrame()

    for ext, clf in clfs.items():
        # Predict
        df_pred_train_ = pd.DataFrame(clf.predict(train), columns=["{}-{}".format(label, ext)])
        df_pred_val_ = pd.DataFrame(clf.predict(val), columns=["{}-{}".format(label, ext)])
        df_pred_test_ = pd.DataFrame(clf.predict(test), columns=["{}-{}".format(label, ext)])
        
        # # Probabilidades
        cols = [
            "{}-{}-{}".format(label, ext, clf.classes_[0]), 
            "{}-{}-{}".format(label, ext, clf.classes_[1]), 
            "{}-{}-{}".format(label, ext, clf.classes_[2])
        ]
        df_prob_train_ = pd.DataFrame(clf.predict_proba(train), columns=cols)
        df_prob_val_ = pd.DataFrame(clf.predict_proba(val), columns=cols)
        df_prob_test_ = pd.DataFrame(clf.predict_proba(test), columns=cols)

        pred_train = pd.concat([pred_train, df_pred_train_], axis=1, sort=False)
        pred_val = pd.concat([pred_val, df_pred_val_], axis=1, sort=False)
        pred_test = pd.concat([pred_test, df_pred_test_], axis=1, sort=False)

        prob_train = pd.concat([prob_train, df_prob_train_], axis=1, sort=False)
        prob_val = pd.concat([prob_val, df_prob_val_], axis=1, sort=False)
        prob_test = pd.concat([prob_test, df_prob_test_], axis=1, sort=False)

    return pred_train, pred_val, pred_test, prob_train, prob_val, prob_test

def main():
    w2v, glove, fasttext = load_embedding()
    for db in ["ZW", "TD", "Union"]:
        if db == 'ZW':
            classe = 'norm'
        else:
            classe = 'class'
        for fold in ["F1", "F2", "F3", "F4", "F5"]:
            # inicializando pred e proba
            df_pred_train = pd.DataFrame()
            df_pred_val = pd.DataFrame()
            df_pred_test = pd.DataFrame()

            df_prob_val = pd.DataFrame()
            df_prob_train = pd.DataFrame()
            df_prob_test = pd.DataFrame()

            # load data
            train = pd.read_csv("./Data/{}/{}/train.csv".format(db, fold))['text']
            val = pd.read_csv("./Data/{}/{}/val.csv".format(db, fold))['text']
            test = pd.read_csv("./Data/{}/{}/test.csv".format(db, fold))['text']
            train_labels = pd.read_csv("./Data/{}/{}/train_labels.csv".format(db, fold))[classe]
            val_labels = pd.read_csv("./Data/{}/{}/val_labels.csv".format(db, fold))[classe]
            class_labels = pd.read_csv("./Data/{}/{}/test_labels.csv".format(db, fold))[classe]

            # Features
            cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
            cv.fit_transform(train)
                
            tfidf =  TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, stop_words='english')
            tfidf.fit_transform(train)

            # SVM
            classifier = get_svm(cv, tfidf, w2v, glove, fasttext)
            clfs = { }

            for ext, clf in classifier.items():
                clfs[ext] = get_classifier(clf['CLF'], train, train_labels, clf['EXT'])
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "SVM", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)
            # LR
            classifier = get_lr(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "SVM", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # RF
            classifier = get_rf(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "RF", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # NB
            classifier = get_nb(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "NB", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # MLP
            classifier = get_mlp(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "MLP", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # Extra
            classifier = get_extra(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "EXTRA", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # KNN
            classifier = get_knn(cv, tfidf, w2v, glove, fasttext)
            # clfs = { }
            for ext, clf in classifier.items():
                clfs.update({ext: get_classifier(clf['CLF'], train, train_labels, clf['EXT'])})
            
            pred_train, pred_val, pred_test, prob_train, prob_val, prob_test = update_pred_proba(clfs, "KNN", train, val, test)
            # Concatenando pred
            df_pred_train = pd.concat([df_pred_train, pred_train], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, pred_val], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, pred_test], axis=1, sort=False)
            # Concatenando prob
            df_prob_train = pd.concat([df_prob_train, prob_train], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, prob_val], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, prob_test], axis=1, sort=False)

            # CNN
            MAX_NB_WORDS = 20000
            MAX_SEQUENCE_LENGTH=300

            y_train = to_categorical(train_labels)
            y_val = to_categorical(val_labels)

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(train)

            seq_train = tokenizer.texts_to_sequences(train)
            seq_val = tokenizer.texts_to_sequences(val)
            seq_test = tokenizer.texts_to_sequences(test)

            data_train = pad_sequences(seq_train, maxlen=MAX_SEQUENCE_LENGTH)
            data_val = pad_sequences(seq_val, maxlen=MAX_SEQUENCE_LENGTH)
            data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)
            # NET
            cnn_cv = get_CNN(cv, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=300, activation='sigmoid', dense=3)
            cnn_tfidf = get_CNN(tfidf, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=300, activation='softmax', dense=3)
            cnn_w2v = get_CNN(w2v, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=300, activation='sigmoid', word_embedding=True, dense=3)
            cnn_glove = get_CNN(glove, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=25, activation='sigmoid', word_embedding=True, dense=3)
            cnn_fast = get_CNN(fasttext, tokenizer, MAX_NB_WORDS, EMBEDDING_DIM=300, activation='sigmoid', word_embedding=True, dense=3)
            # TRAIN
            cnn_cv.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=20)
            cnn_tfidf.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=200)
            cnn_w2v.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=200)
            cnn_glove.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=200)
            cnn_fast.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=200)

            # CONCAT
            # df_pred_test = df_pred_test.drop(['CNN-CV'], axis=1)
            # print(df_pred_test.columns)
            # CV
            cols = ["CNN-CV-0", "CNN-CV-1", "CNN-CV-2"]
            df_train_ = pd.DataFrame(np.argmax(cnn_cv.predict(data_train), axis=1), columns=["CNN-CV"])
            df_val_ = pd.DataFrame(np.argmax(cnn_cv.predict(data_val), axis=1), columns=["CNN-CV"])
            df_test_ = pd.DataFrame(np.argmax(cnn_cv.predict(data_test), axis=1), columns=["CNN-CV"])
            
            df_pred_train = pd.concat([df_pred_train, df_train_], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, df_val_], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, df_test_], axis=1, sort=False)

            # print(df_pred_test.columns)

            # # Probabilidades
            
            df_train_ = pd.DataFrame(cnn_cv.predict(data_train), columns=cols)
            df_val_ = pd.DataFrame(cnn_cv.predict(data_val), columns=cols)
            df_test_ = pd.DataFrame(cnn_cv.predict(data_test), columns=cols)

            df_prob_train = pd.concat([df_prob_train, df_train_], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, df_val_], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, df_test_], axis=1, sort=False)


            cols = ["CNN-TF-0", "CNN-TF-1", "CNN-TF-2"]

            df_train_ = pd.DataFrame(np.argmax(cnn_tfidf.predict(data_train), axis=1), columns=["CNN-TFIDF"])
            df_val_ = pd.DataFrame(np.argmax(cnn_tfidf.predict(data_val), axis=1), columns=["CNN-TFIDF"])
            df_test_ = pd.DataFrame(np.argmax(cnn_tfidf.predict(data_test), axis=1), columns=["CNN-TFIDF"])
            
            df_pred_train = pd.concat([df_pred_train, df_train_], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, df_val_], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, df_test_], axis=1, sort=False)

            # Probabilidades
            
            df_train_ = pd.DataFrame(cnn_tfidf.predict(data_train), columns=cols)
            df_val_ = pd.DataFrame(cnn_tfidf.predict(data_val), columns=cols)
            df_test_ = pd.DataFrame(cnn_tfidf.predict(data_test), columns=cols)

            df_prob_train = pd.concat([df_prob_train, df_train_], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, df_val_], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, df_test_], axis=1, sort=False)

            cols = ["CNN-W2V-0", "CNN-W2V-1", "CNN-W2V-2"]

            df_train_ = pd.DataFrame(np.argmax(cnn_w2v.predict(data_train), axis=1), columns=["CNN-W2V"])
            df_val_ = pd.DataFrame(np.argmax(cnn_w2v.predict(data_val), axis=1), columns=["CNN-W2V"])
            df_test_ = pd.DataFrame(np.argmax(cnn_w2v.predict(data_test), axis=1), columns=["CNN-W2V"])
            
            df_pred_train = pd.concat([df_pred_train, df_train_], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, df_val_], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, df_test_], axis=1, sort=False)

            # Probabilidades
            
            df_train_ = pd.DataFrame(cnn_w2v.predict(data_train), columns=cols)
            df_val_ = pd.DataFrame(cnn_w2v.predict(data_val), columns=cols)
            df_test_ = pd.DataFrame(cnn_w2v.predict(data_test), columns=cols)

            df_prob_train = pd.concat([df_prob_train, df_train_], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, df_val_], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, df_test_], axis=1, sort=False)


            cols = ["CNN-GLOVE-0", "CNN-GLOVE-1", "CNN-GLOVE-2"]

            df_train_ = pd.DataFrame(np.argmax(cnn_glove.predict(data_train), axis=1), columns=["CNN-GLOVE"])
            df_val_ = pd.DataFrame(np.argmax(cnn_glove.predict(data_val), axis=1), columns=["CNN-GLOVE"])
            df_test_ = pd.DataFrame(np.argmax(cnn_glove.predict(data_test), axis=1), columns=["CNN-GLOVE"])

            df_pred_train = pd.concat([df_pred_train, df_train_], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, df_val_], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, df_test_], axis=1, sort=False)

            # Probabilidades
            
            df_train_ = pd.DataFrame(cnn_glove.predict(data_train), columns=cols)
            df_val_ = pd.DataFrame(cnn_glove.predict(data_val), columns=cols)
            df_test_ = pd.DataFrame(cnn_glove.predict(data_test), columns=cols)

            df_prob_train = pd.concat([df_prob_train, df_train_], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, df_val_], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, df_test_], axis=1, sort=False)

            cols = ["CNN-FAST-0", "CNN-FAST-1", "CNN-FAST-2"]

            df_train_ = pd.DataFrame(np.argmax(cnn_fast.predict(data_train), axis=1), columns=["CNN-FAST"])
            df_val_ = pd.DataFrame(np.argmax(cnn_fast.predict(data_val), axis=1), columns=["CNN-FAST"])
            df_test_ = pd.DataFrame(np.argmax(cnn_fast.predict(data_test), axis=1), columns=["CNN-FAST"])

            df_pred_train = pd.concat([df_pred_train, df_train_], axis=1, sort=False)
            df_pred_val = pd.concat([df_pred_val, df_val_], axis=1, sort=False)
            df_pred_test = pd.concat([df_pred_test, df_test_], axis=1, sort=False)

            # Probabilidades
            
            df_train_ = pd.DataFrame(cnn_fast.predict(data_train), columns=cols)
            df_val_ = pd.DataFrame(cnn_fast.predict(data_val), columns=cols)
            df_test_ = pd.DataFrame(cnn_fast.predict(data_test), columns=cols)

            df_prob_train = pd.concat([df_prob_train, df_train_], axis=1, sort=False)
            df_prob_val = pd.concat([df_prob_val, df_val_], axis=1, sort=False)
            df_prob_test = pd.concat([df_prob_test, df_test_], axis=1, sort=False)

            path = "./Data/{}/{}".format(db, fold)
            df_pred_train.to_csv("{}/pred_train.csv".format(path))
            df_pred_val.to_csv("{}/pred_val.csv".format(path))
            df_pred_test.to_csv("{}/pred_test.csv".format(path))

            df_prob_train.to_csv("{}/prob_train.csv".format(path))
            df_prob_val.to_csv("{}/prob_val.csv".format(path))
            df_prob_test.to_csv("{}/prob_test.csv".format(path))


if __name__ == "__main__":
    main()