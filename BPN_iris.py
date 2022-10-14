#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# # 2. BPNN

# ### 2.2 implement: classification problem solved by BPN

# In[7]:


class bpNeuralNetwork():
    # parameter: learning rate& hidden nodes(num); 
    # input num& output num depend on the traing data
    def __init__(self, learning_rate=1e-6, input_num=2, hidden_num=2, output_num=3):
        self.learning_rate = learning_rate
        self.nn_architecture(input_num, hidden_num, output_num)
        self.set_weights()
        
    # traing: feed/back-forward, update weights, transfer the outputs into results, and calculate the accuracy
    def train(self, inputs, true):
        self.feed_forward(inputs)
        self.feed_backward(self.learning_rate, inputs, true)
        acc = self.accuracy(true)
        self.debug()
        return acc
    
    def predict():
        pass
    def data_spilt():
        pass
        
    def nn_architecture(self, input_num, hidden_num, output_num):
        # nodes, input_num, hidden_num, output_num=1
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
    
    def set_weights(self):
        # weights and bias
        self.weights_xh = np.random.normal(scale=1, size=(self.input_num, self.hidden_num))
        self.weights_hy = np.random.normal(scale=1, size=(self.hidden_num, self.output_num))
        self.bias_xh = 1
        self.bias_hy = 1
    
    def sigmoid(self, net):
        return (1+np.exp(-net)) ** -1
    
    def softmax(self, net):
        return np.exp(net)/sum(np.exp(net))
        
    def feed_forward(self, inputs):
        # inputs = [x, y] two dimension
        self.h = self.softmax(np.dot(inputs, self.weights_xh) - self.bias_xh) #dim=(150,2)
        self.y = self.softmax(np.dot(self.h, self.weights_hy) - self.bias_hy) #dim=(150,2)
        return self.y
        
    def feed_backward(self, learning_rate, inputs, true):
        self.delta_y = self.y*(1-self.y)*(true-self.y) #dim=(150,3)
        self.weights_hy += learning_rate*np.dot(self.h.T, self.delta_y) #dim=(2,3)
        self.bias_hy += -learning_rate*self.delta_y.sum()
        
        self.delta_h = self.h*(1-self.h)*np.dot(self.delta_y, self.weights_hy.T) #dim=(150,2)
        self.weights_xh += learning_rate*np.dot(inputs.T, self.delta_h) #dim=(2,2)
        self.bias_xh += -learning_rate*self.delta_h.sum()
        return self.weights_xh, self.bias_xh, self.weights_hy, self.bias_hy
    
    def accuracy(self, true):
        results = []
        for y in self.y:
            results.append(np.argmax(y))
        
        result_encoding = []
        for result in results:
            if result == 0:
                result_encoding.append([1,0,0])
            elif result ==1:
                result_encoding.append([0,1,0])
            else:
                result_encoding.append([0,0,1])
        result_encoding = np.array(result_encoding)
        
        accuracy = []
        for i, result in enumerate(result_encoding):
            if (result == true[i]).all():
                accuracy.append(1)
            else:
                accuracy.append(0)
        acc = sum(accuracy)/len(accuracy)*100
        return acc
            
    def plot(self, data, title):
        fig = plt.figure(figsize=(12,5))
        plt.plot(data, marker='o')
        plt.xlabel('iteration')
        plt.ylabel('accuracy(%)')
        plt.title(title)
        plt.ylim(0,100)
        
    def debug(self):
        print('nodes', self.input_num, self.hidden_num, self.output_num)
        print('h', self.h[0], self.h[-1])
        print('y', self.y[0], self.y[-1])
        print('weights_xh', self.weights_xh[0], self.weights_xh[-1])
        print('weights_hy', self.weights_hy[0], self.weights_hy[-1])
        print('delta_h', self.delta_h[0], self.delta_h[-1])
        print('delta_y', self.delta_y[0], self.delta_y[-1])
        print('ans', np.argmax(self.y[0]), np.argmax(self.y[-1]))


# ### implement

# In[6]:


from sklearn import datasets
from tqdm.notebook import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# define the dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # the first two features only
T = iris.target[:]
one_hot_encoder = OneHotEncoder(sparse=False)
T = one_hot_encoder.fit_transform(np.array(T).reshape(-1, 1))

random.seed(5)
T_list = list(T)
X_list = list(X)
random.shuffle(X_list)
random.shuffle(T_list)

X = np.array(X_list)
T = np.array(T_list)


# In[9]:


nniris = bpNeuralNetwork(learning_rate=0.9, hidden_num=2)

acc = []
for i in tqdm(range(100)):
    print(i)
    results = nniris.train(X, T)
    acc.append(results)
        
nniris.plot(acc, 'iris')


# ### 2.3 implement: forcasting problem solved by BPN

# ### different parameters for comparison to be discussed
# <pre>
# stopping criteria:
# numbers of hidden nodes(hidden_num):
# training rates(learning rate):
# </pre>

# In[ ]:




