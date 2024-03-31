import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import time

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    Z = 1.0/(1.0 + np.exp(-1.0*z))
    return  Z


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    train_pp = np.zeros(shape=(50000, 784))
    validation_pp = np.zeros(shape=(10000, 784))
    test_pp = np.zeros(shape=(10000, 784))
    train_label_pp = np.zeros(shape=(50000,))
    validation_label_pp = np.zeros(shape=(10000,))
    test_label_pp = np.zeros(shape=(10000,))
    
    l_train = 0
    l_valid = 0
    l_test = 0
    l_train_label = 0
    l_valid_label = 0

    for j in mat:

        if "train" in j:
            label = j[-1]
            tuple = mat.get(j)

            ans = range(tuple.shape[0])
            pm1 = np.random.permutation(ans)

            l_tuple = len(tuple) 
            l_tag = l_tuple - 1000 

            train_pp[l_train:l_train + l_tag] = tuple[pm1[1000:], :]
            l_train += l_tag

            train_label_pp[l_train_label:l_train_label + l_tag] = label
            l_train_label += l_tag

            validation_pp[l_valid:l_valid + 1000] = tuple[pm1[0:1000], :]
            l_valid += 1000

            validation_label_pp[l_valid_label:l_valid_label + 1000] = label
            l_valid_label += 1000

        elif "test" in j:

            label = j[-1]
            tuple = mat.get(j)

            ans = range(tuple.shape[0])
            pm1 = np.random.permutation(ans)

            l_tuple = len(tuple)
            test_label_pp[l_test:l_test + l_tuple] = label

            test_pp[l_test:l_test + l_tuple] = tuple[pm1]
            l_test += l_tuple

    size_train = range(train_pp.shape[0])
    pm2 = np.random.permutation(size_train)

    train_data = train_pp[pm2]
    train_data = np.double(train_data)

    train_data = train_data / 255.0
    train_label = train_label_pp[pm2]

    size_valid = range(validation_pp.shape[0])
    vali_perm = np.random.permutation(size_valid)

    validation_data = validation_pp[vali_perm]
    validation_data = np.double(validation_data)

    validation_data = validation_data / 255.0
    validation_label = validation_label_pp[vali_perm]

    size_test = range(test_pp.shape[0])
    test_perm = np.random.permutation(size_test)

    test_data = test_pp[test_perm]
    test_data = np.double(test_data)

    test_data = test_data / 255.0
    test_label = test_label_pp[test_perm]

    # Feature selection
    # Your code here.


    feat = np.array(np.vstack((train_data,validation_data,test_data)))
    n = feat.shape[1]

    col = np.arange(n)
    final = np.all(feat == feat[0,:], axis = 0)

    column_id = col[final]
    sol = np.arange(784)
    
    print('Preprocess Done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label



def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    
    # Your code here
    #
    #
    #
    #
    
    a = np.ones(training_data.shape[0],dtype = np.float64)
    b = np.ones(training_data.shape[0],dtype = np.float64)
    
    training_data = np.column_stack([training_data,a]) 
    x1 = np.dot(training_data,w1.transpose())
    x2 = np.column_stack([sigmoid(x1),b])
    x3 = sigmoid(np.dot(x2,w2.transpose()))
    
    out = np.zeros((len(training_data),10))
    for i in range (0,len(training_data)):
        out[i][int(training_label[i])]=1

    #Error    
    s1 = np.log(x3)     #taking log
    s2 = np.log(1-x3)
    param = np.multiply(out,s1) + np.multiply((1-out),s2)
    sum = np.sum(param)
    error = (sum/(len(training_data)))
    error = np.negative(error)

    #Calculating Gradiant Descent  
    g1 = x3 - out
    grad_w2 = np.dot(g1.transpose(),x2)
    p = ((np.dot(g1,w2))*((1 - x2)*x2))
    grad_w1 = np.dot(p.transpose(),training_data)
    grad_w1 = grad_w1[0:n_hidden,:]

    #Regularization
    w1_2 = np.sum(np.square(w1))
    w2_2 = np.sum(np.square(w2))  
    
    val = (lambdaval/(2*len(training_data)))*(w1_2+w2_2)
    obj_val = error + val
    
    grad_w1 = (grad_w1 + (lambdaval*w1))/len(training_data)
    grad_w2 = (grad_w2 + (lambdaval*w2))/len(training_data)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)



def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    
    labels = np.array([])
    #Your code here

    bias1 = np.ones((data.shape[0],1), dtype = np.uint8)
    data = np.column_stack([data, bias1])

    op = sigmoid(np.dot(data, w1.transpose()))
    bias2 = np.ones((op.shape[0], 1),dtype = np.uint8)
    op = np.column_stack([op, bias2])

    labels = sigmoid(np.dot(op, w2.transpose()))
    labels = np.argmax(labels, axis=1)
    
    return labels



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()



#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.
#opts = {'maxiter': 100}

start_time = time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

end_time = time()
diff = end_time - start_time

print("Training Time: Seconds")
print(diff)


#nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')