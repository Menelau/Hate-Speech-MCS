import nltk
from unicodedata import normalize
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re

def pre_processing(tweet_text):
    # Removendo URLS
    new_text = re.sub(r"http\S+", " ", tweet_text)
    # Removendo RT
    new_text = re.sub('RT @[\w_]+: ', ' ', new_text)
    # Removendo tags
    new_text = re.sub(r"@\S+", " ", new_text)
    # Retirando caracteres especiais
    new_text = normalize('NFKD', new_text).encode('ASCII', 'ignore').decode('ASCII')
    new_text = re.sub('[0-9]', ' ', str(new_text))
    new_text = re.sub('\s+', ' ', new_text)
    return new_text


def tokenize_with_stemmer(sentence):
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in word_tokenize(sentence)]
    return tokens
