# import required packages
import os
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from keras.models import load_model
from utils import load_data, preprocess_sentence


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model
    # loading the saved ANN model
    model = load_model('models/ANN_model_NLP')
    model.load_weights("models/NLP_model_weights.h5")
    
    # loading the save vectorizer
    with open("models/Vectorizer.pickle", "rb") as f:
        CV = pickle.load(f)

	# 2. Load your testing data
    # Importing the testing datasets
    if not 'test_data_NLP.csv' in os.listdir("data"):
        # if csv test data doesnt exist, extract it from files and save it
        X_test, y_test = load_data(False)
        test_data = pd.DataFrame([X_test, y_test]).transpose()
        test_data.columns = ["Review", "Rating"]
        
        # saving the datasets as csv files for ease of loading
        test_data.to_csv("test_data_NLP.csv", index = False)
    
    else:
        # loading the dataset from saved file
        test_data = pd.read_csv("data/test_data_NLP.csv")
        

	# 3. Run prediction on the test data and print the test accuracy
    # preprocessing the reviews in the test data
    test_data['Token'] = test_data['Review'].apply(preprocess_sentence)
    
    
    # Vectorizing using Bag of Words Model
    X_test = CV.transform(test_data.Token).toarray()
    y_test = test_data.Rating
    
    # predicting the test set results
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Evaluating model performance on the test set
    accuracy = accuracy_score(y_pred, y_test)
    F_score = f1_score(y_test, y_pred, average = "weighted")
    CM = confusion_matrix(y_pred, y_test)
    
    print("Test accuracy: ", round(accuracy * 100, 2))
    print("Test F_score: ", round(F_score, 4))
    print("Test confusion matrix: \n", CM)