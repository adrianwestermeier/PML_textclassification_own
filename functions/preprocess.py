import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# used emotions: happiness, sadness, anger, surprise, frustration, neutral, excited
def determine_label(line):
    if line == "happiness":
        return 0
    elif line == "sadness":
        return 1
    elif line == "anger":
        return 2
    elif line == "surprise":
        return 3
    elif line == "frustration":
        return 4
    elif line == "neutral":
        return 5
    else:  # "excited"
        return 6


# we need the max_length in order to determine the input length of the neural net
def determine_max_length(df):
    texts = list()
    try:
        for line in df['text']:
            texts.append(line)
    except KeyError:
        print("key error")
    lengths = list()
    for text in texts:
        lengths.append(len(text.split()))
    maxlen = max(lengths)
    return texts, maxlen


# remove unnecessary symbols
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text.
    # substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    # remove symbols which are in BAD_SYMBOLS_RE from text.
    # substitute the matched string in BAD_SYMBOLS_RE with nothing.

    # stopwords seem to matter so we include them for now
    # text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


# load datadict and assign numerical categories according to labels
def derive_text_and_labels():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    dir_path = os.path.join(dir_path, 'datasets/parsed/iemo_daily_goemotion.csv')
    print(dir_path)
    df = pd.read_csv(dir_path)
    print('df head')
    print(df.head())
    # dataframe = dataframe.reset_index(drop=True)
    try:
        df['text'].apply(clean_text)
        # delete empty strings
        df = df[df['text'] != '']
    except KeyError:
        print('No KeyError text')


    labels = list()
    try:
        for line in df['emotion']:
            # mapping: { 0: happiness, 1: sadness, 2: anger, 3: surprise, 4: frustration, 5: neutral, 6: excited}
            labels.append(determine_label(line))
    except KeyError:
        print('No KeyError emotion')
    texts, maxlen = determine_max_length(df)
    dataframe = df.assign(labels=labels)
    labels = np.array(labels)
    texts = np.array(texts)
    # return texts, labels, maxlen
    return dataframe, maxlen


def create_tokenizer():
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000

    df, _ = derive_text_and_labels()
    # This is fixed.
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer


# create train and test set based on processed raw data
def create_train_test_set(tokenizer, df, maxlen, save_directory=None):
    # transform texts to sentence embeddings
    X = tokenizer.texts_to_sequences(df['text'].values)
    # padding so that all texts have same length
    X = pad_sequences(X, maxlen=maxlen)
    print('Shape of data tensor:', X.shape)
    # transform labels into one-hot vectors
    onehot_labels = tf.keras.utils.to_categorical(df['labels'].to_list(), num_classes=7)
    Y = onehot_labels

    print('Shape of label tensor:', Y.shape)
    print("shape messages: ", X.shape)
    print("shape labels: ", Y.shape)
    print('TEXT MAXLEN = {}'.format(maxlen))

    # split the data into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.25, random_state=42)
    print("X/Y_train shape: ", train_features.shape, train_labels.shape)
    print("X/Y_test shape", test_features.shape, test_labels.shape)

    # save train and test sets for later usage
    output_path = os.path.dirname(os.path.realpath(__file__)) + '/processed_data/'
    np.save('{}x.npy'.format(output_path), train_features)
    np.save('{}y.npy'.format(output_path), train_labels)
    np.save('{}test_x.npy'.format(output_path), test_features)
    np.save('{}test_y.npy'.format(output_path), test_labels)

    # save the word dict for mobile phone usage
    if not save_directory:
        with open('android/word_dict.json', 'w') as file:
            json.dump(tokenizer.word_index, file)
    else:
        with open(save_directory + '/word_dict.json', 'w') as file:
            json.dump(tokenizer.word_index, file)

    return train_features, test_features, train_labels, test_labels


# get a mapping dictionary for emotion-label mapping
def map_number_to_emotion(num):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emotions_dict = pd.read_csv(os.path.join(dir_path, 'datasets/goemotions/emotions_mapping.csv')).to_dict()["emotion"]
    return emotions_dict[int(num)]


def preprocess(save_directory=None):
    df, maxlen = derive_text_and_labels()

    tokenizer = create_tokenizer()

    dataset_tuple = create_train_test_set(tokenizer, df, maxlen, save_directory)

    print('Data preprocessed.')
    return df, tokenizer, maxlen, dataset_tuple


if __name__ == '__main__':
    print('preprocess main')
