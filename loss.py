import tensorflow as tf
import numpy as np
import math

#tensorflow version of firstorder spline

# def y_1(t):
#     y_1_val = 1 + t
#     y_1_val = tf.cast(y_1_val, tf.float32)   
#     return y_1_val


# def y_2(t):
#     y_2_val = 1 - t
#     y_2_val = tf.cast(y_2_val, tf.float32)
#     return y_2_val


# def degt(t):
#     c1 = tf.logical_and(tf.greater_equal(t, -1), tf.less(t, 0))               
#     c2 = tf.logical_and(tf.greater_equal(t, 0), tf.less(t, 1))
#     y  = tf.where(c1, y_1(t), tf.where(c2, y_2 (t), 0.))
#     return y




#tensorflow version of cubic spline

def y_1(t):    
    y_1_val = 2./3. - tf.abs(t)**2 + 0.5*tf.abs(t)**3
    y_1_val = tf.cast(y_1_val, tf.float32)   
    return y_1_val


def y_2(t):    
    y_2_val = (1./6.)*(2. - tf.abs(t))**3
    y_2_val = tf.cast(y_2_val, tf.float32)
    return y_2_val


def degt(t):    
    c1 = tf.less(tf.abs(t), 1) 
    c2 = tf.less(tf.abs(t), 2)
    y  = tf.where(c1, y_1(t), tf.where(c2, y_2 (t), 0.))
    return y


def generateContour(c,value,index):    
    m = 6
    rate = 20
    c = tf.gather(c, index, axis = 1)
    contour = c*value
    contour = tf.reshape(contour,(-1,m*rate,5))
    contour = tf.reduce_sum(contour, axis = -1)    
    return contour


def splinegen2d(c):  
    c = tf.cast(c, tf.float32)
    c_x, c_y = tf.split(c, num_or_size_splits = 2, axis = 1)

    m = 6
    rate = 20

    index_c = []
    for i in range (0,m):
        temp = []
        for j in range (math.floor(i - 2), math.floor(i + 3)):
            if j < 0:
                temp = np.append(temp, m + j)
            if j >= 0 and j <= m - 1:
                temp = np.append (temp, j)
            if j > m - 1:
                temp = np.append(temp, j - m)
        index_c = np.append(index_c, np.tile(temp, rate)).astype(int)

    basis_index = np.array(rate*[2,1,0,-1,-2], dtype = np.float32)
    deltas = np.tile(np.linspace(0,1,rate, endpoint=False, dtype = np.float32), (5,1))
    basis_index += (deltas.T).reshape(*basis_index.shape)
    basis_index = tf.convert_to_tensor(basis_index, dtype = tf.float32)
    value = tf.map_fn(degt, basis_index)
    value = tf.tile(value,(m,))
    contour_x = generateContour(c_x,value,index_c)
    contour_y = generateContour(c_y,value,index_c)
    contour = tf.stack([contour_x, contour_y], axis = -1)    
    return contour
    

def orderedDistance(array1, array2):    
    temp = []       
    for i in range(0,120):
        array2 = tf.roll(array2, [1], [1])
        diff = tf.norm(array1 - array2, axis = -1)
        diff = tf.reduce_sum(diff, axis = -1) 
        temp.append(diff)    
    diffmat = tf.transpose(tf.stack(temp))    
    diffmat = tf.reduce_min(diffmat, axis = -1)    
    return diffmat


def tfCustomDistance(array1, array2):    
    #distance =  0.5*(orderedDistance(array1, array2) + orderedDistance(array2, array1))
    distance =  orderedDistance(array1, array2)     
    return distance


def splineLoss(y_true, c_pred):       
    y_pred = splinegen2d(c_pred)
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)    
    loss = (tfCustomDistance(y_pred, y_true))
    return loss
 