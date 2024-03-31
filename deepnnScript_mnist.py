'''
Comparing single layer MLP with deep MLP (using PyTorch)
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from scipy.io import loadmat
from time import time


# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron():

    class net(nn.Module):
        def __init__(self):
            super().__init__()

            # Network Parameters
            n_hidden_1 = 50  # 1st layer number of features
            #n_hidden_2 = 36  # 2nd layer number of features
            #n_hidden_3 = 256  # 2nd layer number of features
            n_input = 784  # data input
            n_classes = 10

            # Initialize network layers
            self.layer_1 = nn.Linear(n_input, n_hidden_1)
            #self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
            #self.layer_3 = nn.Linear(n_hidden_2, n_hidden_3)
            self.out_layer = nn.Linear(n_hidden_1, n_classes)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            #x = F.relu(self.layer_2(x))
            #x = F.relu(self.layer_3(x))
            x = self.out_layer(x)
            return x

    return net()

# Do not change this
def preprocess():
    # pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    # features = pickle_obj['Features']
    # labels = pickle_obj['Labels']
    # train_x = features[0:21100] / 255
    # valid_x = features[21100:23765] / 255
    # test_x = features[23765:] / 255

    # labels = np.squeeze(labels)
    # train_y = labels[0:21100]
    # valid_y = labels[21100:23765]
    # test_y = labels[23765:]


    mat = loadmat('mnist_all.mat')
    
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

    # train_label = np.zeros((len(train_data),10))
    # for col in range (0,len(train_data)):
    #     train_label[col][int(train_label[col])]=1
    # print(train_label)
    # print(np.ravel(train_label))
    # #train_label = np.ravel(train_label)
    # valid_label = np.zeros((len(valid_data),10))
    # for col in range (0,len(valid_data)):
    #     valid_ylabel[col][int(valid_label[col])]=1
    # test_label = np.zeros((len(test_data),10))
    # for col in range (0,len(test_data)):
    #     test_label[col][int(test_label[col])]=1

    class dataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    trainset = dataset(train_data, train_label)
    validset = dataset(validation_data, validation_label)
    testset = dataset(test_data, test_label)

    return trainset, validset, testset


def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred,y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Construct model
model = create_multilayer_perceptron().to(device)

# Define loss and openptimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# load data
trainset, validset, testset = preprocess()
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
print(enumerate(train_dataloader))

# Training cycle
start_time = time()
for t in range(training_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, cost, optimizer)
print("Optimization Finished!")
test(test_dataloader, model, cost)
end_time = time()
diff = end_time - start_time
print("Training Time: Seconds")
print(diff)