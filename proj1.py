#this project Implements Foldiak's learning rule and 
#demonstrates how it learns a representation that is invariant to translation

#have input vectors in an array
#layer that classifies input vectors
#output that has to be compared to the training data set w/ a loss function 

import pandas as pd
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import string
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

'''
# points a, b
a, b = (0, 1, 0), (1, 0, 1)

# matrix with row vectors of points
A = np.array([a, b])

# 3x3 Identity transformation matrix
I = np.eye(3)

# create the scaling transformation matrix
T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

# create the rotation transformation matrix
T_r = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

color_lut = 'rgbc'
fig = plt.figure()
ax = plt.gca()
xs = []
ys = []
for row in A:
    output_row = I @ row
    x, y, i = output_row
    xs.append(x)
    ys.append(y)
    i = int(i) # convert float to int for indexing
    c = color_lut[i]
    plt.scatter(x, y, color=c)
    plt.text(x + 0.15, y, f"{string.ascii_letters[i]}")
xs.append(xs[0])
ys.append(ys[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()

fig = plt.figure()
ax = plt.gca()
xs_s = []
ys_s = []
for row in A:
    output_row = T_s @ row
    x, y, i = row
    x_s, y_s, i_s = output_row
    xs_s.append(x_s)
    ys_s.append(y_s)
    i, i_s = int(i), int(i_s) # convert float to int for indexing
    c, c_s = color_lut[i], color_lut[i_s] # these are the same but, its good to be explicit
    plt.scatter(x, y, color=c)
    plt.scatter(x_s, y_s, color=c_s)
    plt.text(x + 0.15, y, f"{string.ascii_letters[int(i)]}")
    plt.text(x_s + 0.15, y_s, f"{string.ascii_letters[int(i_s)]}'")

xs_s.append(xs_s[0])
ys_s.append(ys_s[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
plt.plot(xs_s, ys_s, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()

fig = plt.figure()
ax = plt.gca()
for row in A:
    output_row = T_r @ row
    x_r, y_r, i_r = output_row
    i_r = int(i_r) # convert float to int for indexing
    c_r = color_lut[i_r] # these are the same but, its good to be explicit
    letter_r = string.ascii_letters[i_r]
    plt.scatter(x_r, y_r, color=c_r)
    plt.text(x_r + 0.15, y_r, f"{letter_r}'")

plt.plot(xs, ys, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()
'''

#n is the number of dimensions 
#returns a list containing the components of an unbiased unit vector
def randomvector(n):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return v

print(' '.join(map(str, randomvector(2)))) 

#sample dataset using iris 
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
print(df)

#splitting datset into training and testing data (80% to 20%)
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


inputs = [[1, 1], [1, -1], [-1, 1], [-1,-1]]

input_vectors = []
def input(n_inputs) :
    for i in range(n_inputs) :
        input_vectors.append(randomvector(2))

#layer function 
'''
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.zeros((2, n_inputs))
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if np.all(self.output == 0):
            self.biases += 1

'''

w1 = w2 = b = 0
d_b = []
d_w = []
y = 1
threshold = 1

for i in (input_vectors) : 
    d_b.append(b)
    d_w.append([w1, w2])
    w1 = w1 + (input_vectors[i][0] * y)
    w2 = w2 + (input_vectors[i][1] * y)
    b = b + y

net_input = np.dot(input_vectors, d_w) + b

def activation_function(net_input):
    if net_input >= threshold:
        return 1
    elif -threshold <= net_input & net_input <= threshold:
        return 0
    else:
        return -1

M1 = [
    0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
    ]

M2 = [
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
    ]


M3 = [
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
    ]

M4 = [
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
    ]

M5 =[
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
]

random_int = random.sample(range(64), 64)
c_unit1_sum = 0
c_unit2_sum = 0
c_unit3_sum = 0
c_unit4_sum = 0

c_unit1 = random_int[0:16]
c_unit2 = random_int[16:32]
c_unit3 = random_int[32:48]
c_unit4 = random_int[48:64]

for i in (64):
    while i < 16:
        c_unit1_sum += M1[i] 
    while 16 <= i < 32: 
        c_unit2_sum += M1[i] 
    while 32 <= i < 48:
        c_unit3_sum += M1[i] 
    while 48 <= i < 64: 
         c_unit4_sum += M1[i] 
    
activated_unit = max(c_unit1_sum, c_unit2_sum, c_unit3_sum, c_unit4_sum)
