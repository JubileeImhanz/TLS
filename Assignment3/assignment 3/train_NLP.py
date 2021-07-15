# import required packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from utils import load_data, preprocess_sentence, word_vectorizer, accuracy_loss_plot, print_final_result
import pickle


if __name__ == "__main__": 
    
	# 1. load your training data
    
    """ Importintg the training data """
    if not os.path.isfile("train_data_NLP.csv"):
        # if csv taining data doesnt exist, extract it from files and save it
        X_train, y_train = load_data()
        train_data = pd.DataFrame([X_train, y_train]).transpose()
        train_data.columns = ["Review", "Rating"]
        
        # saving the datasets as csv files for ease of loading
        train_data.to_csv("train_data_NLP.csv", index = False)
    
    else:
        # loading the dataset from saved file
        train_data = pd.read_csv("train_data_NLP.csv")

	# 2. Train your network
	
    """ Data Preprocessing """
    # preprocessing the reviews in the training data
    train_data['Token'] = train_data['Review'].apply(preprocess_sentence)
        
    # creating the word embedding using count vectorizer
    X, CV = word_vectorizer(train_data.Token, "CountVectorizer", 5000)
    y = train_data.Rating
    
    # splitting the data into training and validatuion set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=0)
    
    """ Building the ANN model """
    model = Sequential()
    
    # Input - Layer
    model.add(Dense(128, activation = "relu", input_shape=(X_train.shape[1], )))
    
    # Hidden - Layers
    model.add(Dropout(0.3, seed=None))
    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.2, seed=None))
    model.add(Dense(16, activation = "relu"))
    
    # Output- Layer
    model.add(Dense(1, activation = "sigmoid"))
    model.summary()
    
    #Compiling the ANN
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # fitting the ANN to the training set
    history = model.fit(X_train,y_train,
                        epochs=40,
                        validation_data=(X_val, y_val),
                        verbose=1, # print result every epoch
                        batch_size=128)

    # plotting the training and validation accuracy and loss for every epoch 
    accuracy_loss_plot(history)
    
    # printing the final model accuracy and loss
    print_final_result(history)

	# 3. Save your model
    # save the neural network model
    model.save("ANN_model_NLP")
    model.save_weights("NLP_model_weights.h5")
        
    # save the vectorizer
    with open("Vectorizer.pickle", "wb") as f:
        pickle.dump(CV, f)

    