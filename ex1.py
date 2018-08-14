import numpy as np

def sigmoid_activation(z):
    s = 1 / (1 + exp(-z))
    return(s)
    
def compute_cost(X, Y, W):
    m = Y.size;
    H = h(X,W)
    error = H-Y
    J = (1/(2*m)) *  np.sum(error **2, axis=0)
    
    return J

def h(X, W):
    return np.dot(X, W)

def batch_gradient_descent(X, Y , W, alpha, num_iters):
    m = len(Y)
    J_history = np.zeros((num_iters,1))
    # in each iteration, perform a single gradient step on the weights vector
    for i in range(num_iters):        
        H = h(X,W)
#        W = W - (alpha/m) * np.sum((H - Y) * X) # update weights - INCORRECT
        error = H - Y 
        W = W - (alpha/m) * np.dot(X.T,error) # update weights
        J_history[i] = compute_cost(X, Y, W)
#        print("cost=", J_history[i], "W=", W)
    return W, J_history    

# load data
dataset = np.loadtxt('ex1data1.csv', delimiter=',', skiprows=1)

# split dataset into X and Y (input and labels)
X = dataset[:,0:1]
Y = dataset[:,1]
Y = Y.reshape((Y.size,1)) # => it is NECESSARY to reshape Y to have a shape with column 1, otherwise the results will be INCORRECT.
m = Y.size

# to take into account the intercept term (w0), add a column of ones to X as the intercept column (first column). This allows w0 to be treated simply as another feature.
ones = np.ones((X.shape[0],1))
#X = np.hstack((ones,X)) # either this method or the below method to insert a column of ones can work
X = np.insert(X,0,1, axis=1) 

# initialize weights (to 0), number of iterations and the learning rate
W = np.zeros((X.shape[1],1))
iterations=1500
alpha=0.01

# compute cost at iteration zero for validating the correctness of code
cost = compute_cost(X, Y, W)
print("cost at iteration 0=", cost)

# now train the network by using gradient descent
W, J_history = batch_gradient_descent(X, Y, W, alpha, iterations)

# print final values of cost and the weights
print("final cost=", J_history[-1]) 
print("final weights=", W)

# now make predictions with the weights determined by gradient descent
# prediction for area with 35,000 population
prediction_1 = h([1,3.5],W) * 10000
print("prediction_1=", prediction_1)
# prediction for area with 70,000 population
prediction_2 = h([1,7],W) * 10000
print("prediction_2=", prediction_2)
