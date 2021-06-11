import numpy as np
from sklearn.model_selection import train_test_split

class Neural_Network(object):
    """ 
    Class of a Neural network with 1 input layer, 1 hidden layer and 1 output layer
    """
    def __init__(self, input_size = 2, hidden_size = 2, output_size = 1, 
                 activation = "sigmoid", epochs = 1000, alpha = 0.001):
        """
        input_size = no of input features
        hidden_size = no of neurond in hidden layer
        output_size = no of classes for classification problem
        alpha = learning rate
        """
        
        # setting the number on neurons in each layer
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        self.epochs = epochs # no of training epochs
        self.alpha = alpha # learning rate
        
        # setting the default activation function
        self.activation = self.sigmoid
        self.activation_gradient = self.sigmoid_gradient
        
        # setting the activation function            
        if activation == 'tanh':
            self.activation = self.tanh
            self.activation_gradient = self.tanh_gradient
      
        # Randomly initializing the weights (-1 to 1) and adding 1 to represent the bias      
        # weight matrix from input to hidden layer
        self.W1 = 2 * np.random.random((self.inputSize + 1, self.hiddenSize + 1)) - 1 
        
        # weight matrix from hidden to output layer
        self.W2 = 2 * np.random.random((self.hiddenSize + 1, self.outputSize)) - 1 


    def forward(self, X):
        #forward propagation through the network
        self.z1 = np.dot(X, self.W1) # dot product of X (input) and W1 (input - hidden weights)
        self.z2 = self.activation(self.z1) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer output and hidden - output weights
        o = self.activation(self.z3) # activation function to get predicted output
        return o 
      
        
    def backward(self, X, y, o):
        # backward propgation through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.activation_gradient(o) # applying activation derivative to error
        
        self.z2_error = self.o_delta.dot(self.W2.T) # effect of hidden layer weights on output error
        self.z2_delta = self.z2_error * self.activation_gradient(self.z2) # applying activation derivative to z2 error
        
        self.W1 += self.alpha * (X.T.dot(self.z2_delta)) # updating (input - hidden) weights
        self.W2 += self.alpha * (self.z2.T.dot(self.o_delta)) # updating (hidden - output) weights
        
        
    def fit(self, X, y):
        # Add column of ones to X to represent the bias unit of the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for i in range(self.epochs): # trains the NN 'epochs' times
            if i % 50 == 0:
                print("Epoch: ", i)
            o = self.forward(X)
            self.backward(X, y, o)
        
        self.model_loss = self.calc_model_loss(X, y)
    
            
    def sigmoid(self, z):
        # sigmoid function
        return 1 / ( 1 + np.exp(-z))
    
        
    def sigmoid_gradient(self, z):
        #derivative of sigmoid
        return z * (1 - z)
    
    
    def tanh(x):
        # tanh function
        return np.tanh(x)
    
    
    def tanh_gradient(x):
        # derivative of tanh function
        return 1.0 - x**2
    
            
    def get_model(self):
        # returns the learned weights of the model
        return [self.W1, self.W2]
     
    
    def predict_proba(self, x_new): 
        # returns the probability of input belonging to the different classes
        ones = np.atleast_2d(np.ones(x_new.shape[0]))
        x_new = np.concatenate((ones.T, x_new), axis=1)
        
        y_pred_proba = self.forward(x_new)
        
        # for multiclass outputs, sum of classes probability must be 1
        if self.outputSize != 1:
            scale = y_pred_proba.sum(axis = 1).reshape(-1,1)
            y_pred_proba = y_pred_proba / scale
        
        return y_pred_proba
    
    
    def predict(self, x_new): 
        # returns the model's class prediction of given input
        y_pred = self.predict_proba(x_new)
        
        # for binary classification set threshold = 0.5
        if self.outputSize == 1:
            y_pred = (y_pred >= 0.5).astype(int)
         
        # for multiclass classification, assign to class with highest probability
        else:
            y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
            
        return y_pred
    
    
    def calc_model_loss(self, X, y):
        # mean sum squared loss
        return np.mean(np.square(y - NN.forward(X)))
    
    
    def save_model(self, filename):
        # saves model weights in specified filename
        filename = filename + ".npy"
        arr = np.array([self.W1, self.W2], dtype="object")
        np.save(filename, arr, allow_pickle=True)
        
    
    def load_weights(self, weight):
        # load weights into neural network
        self.W1 = weight[0]
        self.W2 = weight[1]
        
    

def accuracy(y_true, y_pred):
    if not (len(y_true) == len(y_pred)):
        print('Size of predicted and true labels not equal.')
        return 0.0

    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_pred[i]).all() else 0

    return corr/len(y_true)


if __name__ == '__main__':
    
    # Importing the dataset
    X = np.loadtxt(open("train_data.csv", "rb"), delimiter=",", skiprows=0)
    y = np.loadtxt(open("train_labels.csv", "rb"), delimiter=",", skiprows=0)
    
    # Splitting the data into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=275)
    
            
    
    # Initializing the Neural network
    NN = Neural_Network(X.shape[1], 10, y.shape[1], epochs = 500)
    
    # Fitting the network to the train set
    NN.fit(X_train, y_train)
    
    # Checking the performance of the model on training set
    y_pred_train = NN.predict(X_train)
    print("Training Accuracy: ", accuracy(y_train,y_pred_train) * 100)
    print("Model loss: ", NN.model_loss)
    
    # Checking the performance of the model on test set (unseen observations)
    y_pred_val = NN.predict(X_val)
    print("Test Accuracy: ", accuracy(y_val, y_pred_val) * 100)
    
    
    NN.save_model("NN_model")
    np.savetxt("X_val.csv", X_val, delimiter=",")
    np.savetxt("y_val.csv", y_val, delimiter=",")
    