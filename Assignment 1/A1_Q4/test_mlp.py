import numpy as np

STUDENT_NAME = 'Jubilee Imhanzenobe, Olohireme Ajayi, Hanoor Singh'
STUDENT_ID = '20809735, 20869827, 20870613'

def test_mlp(data_file):
    # Import libraries
    from back_prop_mat import Neural_Network
    
    # Load the test set
    # START
    X_test = np.loadtxt(open(data_file, "rb"), delimiter=",", skiprows=0)
    # END

    # Load your network
    # START
    weights = np.load("NN_model_weights.npy", allow_pickle = True) 
    NN = Neural_Network()
    NN.load_weights(weights)
    
    # END


    # Predict test set - one-hot encoded
    # y_pred = ...
    y_pred = NN.predict(X_test)

    # return y_pred
    return y_pred

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''




