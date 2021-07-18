import os
import re
import glob
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt



STOPWORDS = set(stopwords.words('english'))

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def _read_text_file(path):
    with open(path, 'rt',encoding="utf8") as file:
        lines = file.readlines()
        text = " ".join(lines)
    return text

def load_data(train = True):
    
    train_test_path = "train" if train else "test"

    dir_base = os.path.join("aclImdb", train_test_path)

    path_pattern_pos = os.path.join(dir_base, "pos", "*.txt")
    path_pattern_neg = os.path.join(dir_base, "neg", "*.txt")

    paths_pos = glob.glob(path_pattern_pos)
    paths_neg = glob.glob(path_pattern_neg)

    data_pos = [_read_text_file(path) for path in paths_pos]
    data_neg = [_read_text_file(path) for path in paths_neg]

    x = data_pos + data_neg

    y = [1.0] * len(data_pos) + [0.0] * len(data_neg)

    return x, y

def w2v_preprocess(sentence):
    """
    funtion that formats string to remove special characters
    """
    # remove invalid and special characters
    review = re.sub('[^a-z A-Z]', '', sentence)
    
    # tokenize the sentence
    token = nltk.word_tokenize(review.lower())
    lem = WordNetLemmatizer()
    lemmatized_token = [lem.lemmatize(word, 'v') for word in token if not word in STOPWORDS]
    return lemmatized_token


def preprocess_sentence(sentence, stemmer = "PS"):
    """
    funtion that formats string to remove special characters
    """
    # remove invalid and special characters
    review = re.sub('[^a-z A-Z]', '', sentence)
    
    # tokenize the sentence
    token = nltk.word_tokenize(review.lower())
    
    # Reduce words to their root word by stemming or lemmatizing 
    # and remove stopwords
    if stemmer == "PS":
        ps = PorterStemmer()
        stemmed_token = [ps.stem(word) for word in token if not word in STOPWORDS]
        return " ".join(stemmed_token)
    else:
        lem = WordNetLemmatizer()
        lemmatized_token = [lem.lemmatize(word, 'v') for word in token if not word in STOPWORDS]
        return " ".join(lemmatized_token)


def word_vectorizer(data, method, max_features = None):
    if method == "CountVectorizer":
        vectorizer = CountVectorizer(max_features = max_features)
        X = vectorizer.fit_transform(data).toarray()
    else:
        vectorizer = TfidfVectorizer(max_features = max_features, norm='l2')
        X = vectorizer.fit_transform(data).toarray()
        
    return X, vectorizer


def get_sentence_embedding(data, model):
    sentence_embeddings = []
    for token in data:
        embeddings = []
        for word in token:
            try:
                embeddings.append(list(model.wv.__getitem__(word)))
            except:
                continue
        
        embedding_array = np.array(embeddings)
        sentence_embedding = np.mean(embedding_array, axis=0)
        sentence_embeddings.append(list(sentence_embedding))
        
    features = len(sentence_embeddings[0])
    df = pd.DataFrame(sentence_embeddings, columns = ["feature_"+ str(i+1) for i in range(features)])
    return df   


def accuracy_loss_plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='lower right')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper right')
    
    
def print_final_result(history):
    print("\n", "*" * 20, "Training set Evaluation", "*" * 20)
    print("Final Train accuracy: ", round(history.history["accuracy"][-1], 4))
    print("Final Train loss: ", round(history.history["loss"][-1], 4))
    
    print("\n", "*" * 20, "Validation set Evaluation", "*" * 20)
    print("Final Val accuracy: ", round(history.history["val_accuracy"][-1], 4))
    print("Final Val loss: ", round(history.history["val_loss"][-1], 4))
    

def compare_models(history_list, labels):
    """ Function for plotting the accuracy and loss for training and validation set to compare the models """
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    count = 0
    epochs = list(range(1, 41))
    final_train = []
    final_val = []
    final_loss = []
    final_val_loss = []
    for history in history_list:
        label = labels[count]
        val_accuracy = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        train_loss = history.history['loss']
        axs[0,0].plot(epochs, train_accuracy, label= label)
        axs[1,0].plot(epochs, train_loss, label=label)
        axs[0,1].plot(epochs, val_accuracy, label= label)
        axs[1,1].plot(epochs, val_loss, label=label)
        count += 1
        
        final_train.append(round(history.history['accuracy'][-1] * 100, 2))
        final_val.append(round(history.history['loss'][-1], 4))
        final_loss.append(round(history.history['val_accuracy'][-1] * 100, 2))
        final_val_loss.append(round(history.history['val_loss'][-1], 4))

    axs[0,0].set_ylabel('Tarining accuracy')
    axs[1,0].set_ylabel('Training loss')
    axs[0,1].set_ylabel('Validation accuracy')
    axs[1,1].set_ylabel('Validation loss')

    axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.1, -1.4),
              ncol=4, fancybox=True, shadow=True)
    
    columns = ["Train Accuracy", "Train Loss", "Val Accuracy", "Val Loss"]
    result = pd.DataFrame(np.array([final_train, final_val, final_loss, final_val_loss]).T, index = labels, columns = columns)
    return result
        