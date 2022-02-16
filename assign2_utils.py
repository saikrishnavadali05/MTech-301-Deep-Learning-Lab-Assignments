import numpy as np
import h5py

def load_train_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # train images
    train_y = np.array(train_dataset["train_set_y"][:]) # train labels
    return train_x, train_y
    
def load_test_data():
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # test images
    test_y = np.array(test_dataset["test_set_y"][:]) # test labels
    return test_x, test_y

def flatten(z):
    m, h, w, c = z.shape
    nx = h*w*c
    z = z.reshape((m, nx))
    return z.T # 'T' is for transpose

def initialize_params(nx):
    """
      Function that intializes weights to scaled random std normal values and biases to zero and returns them
      
      nx: a list that contains number of nodes in each layer in order. For a l-layer network, len(nx) = l+1 
          as it includes num of features in input layer also.
          
      returns W: list of numpy arrays of weight matrices
              b: list of numpy arrays of bias vectors
    """
    Wlist = []
    blist = []
    for i in range(1, len(nx)): 
        Wlist.append(np.random.randn(nx[i], nx[i-1]) * 0.01) # np.random.randn will be useful
        blist.append(np.zeros((nx[i], 1))) # np.zeros will be useful
    return Wlist, blist

def f(z, fname = 'ReLU'):
    """
      computes and returns the non-linear function of z given the non-linearity
      
      z: numpy array of any shape on which the non-linearity will be applied elementwise
      fname: a string that is name of the non-linearity. Defaults to 'ReLU'. Other valid values are
             'Sigmoid', 'Tanh', and 'Linear'.
      
      returns f(z) f is the non-linear function whose name is fname
    """
    if fname == 'ReLU':
        return np.maximum(z, 0)
    elif fname == 'Sigmoid':
        return 1./(1+np.exp(-z))
    elif fname == 'Tanh':
        return np.tanh(z)
    elif fname == 'Linear':
        return z
    else:
        raise ValueError('Unknown non-linear function error')
        
def df(z, fname = 'ReLU'):
    """
      computes and returns the derivative of the non-linear function of z with respect to z
      
      z: numpy array of any shape 
      fname: a string that is name of the non-linearity. Defaults to 'ReLU'. Other valid values are
             'Sigmoid', 'Tanh', and 'Linear'.
      
      returns df/dz where f is the non-linear function of z. Name of the non-linear function is fname.
    """
    if fname == 'ReLU':
        return z>0
    elif fname == 'Sigmoid':
        sigma_z = 1./(1+np.exp(-z))
        return sigma_z * (1-sigma_z)
    elif fname == 'Tanh':
        return 1 - np.tanh(z)**2
    elif fname == 'Linear':
        return np.ones(z.shape)
    else:
        raise ValueError('Unknown non-linear function error')

def forward(a, W, b, fname = 'ReLU'):
    """
      Forward propagates a through the current layer given W and b
      a: I/p activation from previous activation layer l-1 of shape nx[l-1] x m
      w: weight matrix of shape nx[l] x nx[l-1]
      b: bias vector of shape nx[l+1] x 1
      
      returns anew: the output activation from current layer of shape nx[l] x m
              cache: a tuple that contains current layer's linear computation z, previous layer's activation a,
                     current layer's activation anew and weight matrix W
    """
    z = W @ a + b              # np.dot or np.matmul or @ operator will be useful. Also understand numpy 
                               # broadcasting for adding vector b to product of W and a
    anew = f(z, fname)         # function f defined aboove will be useful
    cache = (z, a, anew, W)    # read the doc string for this function listed above and acoordingly fill rhs
    return anew, cache
        
def update_params(Wlist, blist, dWlist, dblist, alpha):
    """
      Updates all the parameters using gradient descent rule
      
      Wlist: a list of all weight matrices to be updated
      blist: a list of bias vectors to be updated
      dWlist: a list of gradients of loss with respect to weight matrices
      dblist: a list of gradients of loss with respect to bias vectors
      alpha: learning rate
    """
    for i in range(len(Wlist)):
        Wlist[i] -= alpha*dWlist[i] # fill rhs
        blist[i] -= alpha*dblist[i] # fill rhs
        
def predict(a, Wlist, blist, fname_list):
    for l in range(len(fname_list)):
            a, _ = forward(a, Wlist[l], blist[l], fname_list[l])
    predictions = np.zeros_like(a)
    predictions[a > 0.5] = 1
    return predictions

def test_model(a, y, Wlist, blist, fname_list):
    predictions = predict(a, Wlist, blist, fname_list)
    acc = np.mean(predictions == y)
    acc = np.asscalar(acc)
    return acc