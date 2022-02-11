# %% 
import numpy as np

#diagonal sweeps
r0 = [0,0,0,0,0,0,0,0]
r1 = [1,0,0,0,0,0,0,0]
r2 = [0,1,0,0,0,0,0,0]
r3 = [0,0,1,0,0,0,0,0]
r4 =[0,0,0,1,0,0,0,0]
r5 =[0,0,0,0,1,0,0,0]
r6 =[0,0,0,0,0,1,0,0]
r7 =[0,0,0,0,0,0,1,0]
r8 = [0,0,0,0,0,0,0,1]

rLine = [1,1,1,1,1,1,1,1]
rEmpty = [0,0,0,0,0,0,0,0]

rA = [0,0,0,1,1,0,0,0]
rB = [0,0,1,1,1,1,0,0]

#from middle up right
diagonal_sweep1 =[]

matrix = r1,r2,r3,r4,r5,r6,r7,r8
matrix1 = np.asarray(matrix)
for row in matrix:
    matrix1 = np.delete(matrix1, 0, axis=0)
    matrix1 = np.r_[matrix1, [r0]]
    diagonal_sweep1.append(matrix1)

#from up right down
diagonal_sweep2 = diagonal_sweep1.copy()
diagonal_sweep2 = np.flip(diagonal_sweep2, axis=0)

#from middle down right
diagonal_sweep3 = diagonal_sweep1.copy()
diagonal_sweep3 = np.flip(diagonal_sweep3, axis=1)

#from down right up
diagonal_sweep4 = diagonal_sweep2.copy()
diagonal_sweep4 = np.flip(diagonal_sweep4, axis=1)

#from down left up
diagonal_sweep5 = diagonal_sweep1.copy()
diagonal_sweep5 = np.flip(diagonal_sweep5)

#from middle down left
diagonal_sweep6 = diagonal_sweep2.copy()
diagonal_sweep6 = np.flip(diagonal_sweep6)

#from top left down
diagonal_sweep7 = diagonal_sweep3.copy()
diagonal_sweep7 = np.flip(diagonal_sweep7)

#from middle left up
diagonal_sweep8 = diagonal_sweep4.copy()
diagonal_sweep8 = np.flip(diagonal_sweep8)

#full sweep from top right corner to bottom left
TopRBottomL = []
for matrix in diagonal_sweep2:
    TopRBottomL.append(matrix)
for matrix in diagonal_sweep6:
    TopRBottomL.append(matrix)

#full sweep from bottom right corner to top left
BottomRTopL = []
for matrix in diagonal_sweep4:
    BottomRTopL.append(matrix)
for matrix in diagonal_sweep8:
    BottomRTopL.append(matrix)

# full sweep top left to bottom right
TopLBottomR = []
for matrix in diagonal_sweep7:
    BottomRTopL.append(matrix)
for matrix in diagonal_sweep3:
    BottomRTopL.append(matrix)

# full sweep from bottom left corner to top right
BottomLTopR = []
for matrix in diagonal_sweep5:
    BottomLTopR.append(matrix)
for matrix in diagonal_sweep1:
    BottomLTopR.append(matrix)

#vertical sweeps

#down to up
vertical_sweep1 = []
matrix = rEmpty, rEmpty, rEmpty, rEmpty, rEmpty, rEmpty, rEmpty, rLine
matrix2 = np.asarray(matrix)
for row in matrix:
    matrix2 = np.delete(matrix2, 0, axis=0)
    matrix1 = np.r_[matrix2, [rEmpty]]
    vertical_sweep1.append(matrix2)

#up to down
vertical_sweep2 = vertical_sweep1.copy()
vertical_sweep2 = np.flip(vertical_sweep2, axis=0)

#horisontal sweeps
horisontal_sweep1 = []
for matrix in vertical_sweep1:
    matrix3 = matrix.T
    horisontal_sweep1.append(matrix3)

horisontal_sweep2 = []
for matrix in vertical_sweep2:
    matrix4 = matrix.T
    horisontal_sweep1.append(matrix4)

#rotation
rotation_1 =[]

matrix = r1,r2,r3,r4,r5,r6,r7,r8
first_matrix = np.asarray(matrix)
matrix5 = np.asarray(matrix)
rotation_1.append(first_matrix)
i = 0
while i< 4:
    matrix5[:i+1] = matrix[1+i]
    matrix5[7-i:] = matrix[6-i]
    new_value = matrix5.copy()
    rotation_1.append(new_value)
    i += 1

item6 = np.flip(rotation_1[2], axis=1)
item7 = np.flip(rotation_1[1], axis=1)
item8 = np.flip(rotation_1[0], axis = 1)

rotation_1.append(item6)
rotation_1.append(item7)
rotation_1.append(item8)

#scale change

scale_change1 =[]
matrix = rEmpty,rEmpty,rEmpty,rEmpty,rLine,rEmpty,rEmpty,rEmpty
matrix = np.asarray(matrix)
matrix6 = rEmpty,rEmpty,rEmpty,rLine,rLine,rEmpty,rEmpty,rEmpty
matrix6 = np.asarray(matrix6)
matrix7 = rEmpty,rEmpty,rLine,rLine,rLine,rEmpty,rEmpty,rEmpty
matrix7 = np.asarray(matrix7)

i = 0
while i < 2:
    scale_change1.append(matrix)
    scale_change1.append(matrix6)
    scale_change1.append(matrix7)
    scale_change1.append(matrix6)
    i += 1

scale_change2 = []
matrix = rEmpty,rEmpty,rEmpty,rEmpty,rA,rEmpty,rEmpty,rEmpty
matrix = np.asarray(matrix)
matrix8 = rEmpty,rEmpty,rEmpty,rB,rEmpty,rEmpty,rEmpty,rEmpty
matrix8 = np.asarray(matrix8)
matrix9 = rEmpty,rEmpty,rEmpty,rLine,rEmpty,rEmpty,rEmpty,rEmpty
matrix9 = np.asarray(matrix9)
i = 0
while i < 2:
    scale_change2.append(matrix)
    scale_change2.append(matrix8)
    scale_change2.append(matrix9)
    scale_change2.append(matrix8)
    i += 1
#______________________________________________________

#collapse matrix into array 
#(now we can refer to each element aka simple unit individually)
simple_units = np.asarray(matrix).flatten()
 
#initial weights for each simple cell-unit pair sampled from uniform distribution
##weights = #simple units
w1j = np.random.uniform(0, 0.1, len(simple_units))
w2j = np.random.uniform(0, 0.1, len(simple_units))
w3j = np.random.uniform(0, 0.1, len(simple_units))
w4j = np.random.uniform(0, 0.1, len(simple_units))

unit_sum = []
for i in range(len(simple_units)):
    unit_sum.append(0.2)

def complex_unit(wij): 
    for i in range(len(simple_units)):
        unit_sum[i] = (simple_units[i] * wij[i])
    return unit_sum

#compute which complex unit will be activated
c1 = sum(complex_unit(w1j))
c2 = sum(complex_unit(w2j))
c3 = sum(complex_unit(w3j))
c4 = sum(complex_unit(w4j))

#get index of max element (corresponds to which unit is activated)
sum_list = [c1, c2, c3, c4]
max_sum = max(sum_list)
max_index = sum_list.index(max_sum)

#y is for every unit not every simple cell
yt_prev = []
yt = []
y_avg = []

#activate one of the complex units
def unit_activation():
    for i in range (4):
        if i == max_index:
            yt[i] = 1
        else:
            yt[i] = 0

#compute y(t) 
def y_bar(delta):
    for i in range (4):
        if yt_prev == []:
            y_avg[i] = delta * yt[i]
        else:
            y_avg[i] = (1 - delta)*(yt_prev[i]) + (delta)*(yt[i])
    yt_prev = y_avg

#implement weight updating   
def model(complex_unit, index, alpha):
    weight_change = []
    for xj, wij in zip(simple_units, complex_unit):
        weight_change.append(alpha * y_avg(index) * (xj-wij))
    for delta_wij, wij in zip(weight_change, complex_unit):
        complex_unit.append(delta_wij + wij)

def training():
    #500 is just trials 
    for i in range (50):
        unit_activation()
        y_bar(0.2)
        model(w1j, 0, 0.02)
        model(w2j, 1, 0.02)
        model(w3j, 2, 0.02)
        model(w4j, 3, 0.02)
        synap_weights1.append(w1j)
        synap_weights2.append(w2j)
        synap_weights3.append(w3j)
        synap_weights4.append(w4j)

inputs = []
outputs = []
traces = []
synap_weights1 = []
synap_weights2 = []
synap_weights3 = []
synap_weights4 = []

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
data2D1 = synap_weights1 
data2D2 = synap_weights2
data2D3 = synap_weights3
data2D4 = synap_weights4 
im = plt.imshow(data2D, cmap="copper_r")
plt.colorbar(im)
plt.show()

import matplotlib.pyplot as plt

plt.rc('grid', linestyle="-", color='black')
plt.scatter(y_avg[0], y_avg[1], y_avg[2], y_avg[3], color=['red', 'green', 'blue', 'pink'])
plt.grid(True)

plt.show()
    
# %%
#graph that shows all the changes and then pick like a random 5
#scatterplot of weight changes
    #x should be trials, y should be the value of the weight
    #make 4 classes aka each of the complex units 
#scatterplots of inputs and outputs
    #inputs are the data
    #explain how this works 
#y-traces--show at multiple time points the differences traces for the uints 
    #over time it should be that the same complex unit