import numpy as np
import tensorflow as tf


def initialize_uninitialized(sess):

    with sess.graph.as_default() as g:
        global_vars = tf.global_variables()
        initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, initialized) if not f]

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))


def __bias__(root_name, shape):
    init = tf.zeros_initializer()
    return tf.get_variable(name =  root_name + '_bias', shape = shape, initializer = init, trainable = True)


def __weight__(root_name, shape):
    size = 1.
    for d in shape:
        size *= d
    size = float(size)
    init =  tf.random_normal_initializer(mean = 0.0, stddev = 1./tf.sqrt(size), seed = 0)
    return tf.get_variable(name = root_name + '_weight', shape = shape, initializer = init, trainable = True)

    
def __Activation__(x, activation = None):
    
    if activation is 'sigmoid':
        x = tf.nn.sigmoid(x)
    elif activation is 'relu':
        x = tf.nn.relu(x)
    elif activation is 'tanh':
        x = tf.nn.tanh(x)
    elif activation is 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    elif activation is 'selu':
        x = tf.nn.selu(x)
    
    return x
    

def __Dense__(x, dim_out, activation = None):
    
    dim_in = int(x.shape[-1])
    W = __weight__(root_name = 'dense', shape = [dim_in,dim_out])
    b = __bias__(root_name = 'dense', shape = [dim_out])
    
    x = tf.matmul(x,W) + b
    x = __Activation__(x, activation)
    
    return x
    
 
def __Conv2D__(x, channels_out, filter_dim, activation = None):

    W = __weight__(root_name = 'conv2d', shape = [filter_dim, filter_dim, int(x.shape[-1]), channels_out])
    b = __bias__(root_name = 'conv2d', shape = [channels_out])

    x = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME') + b
    x = __Activation__(x, activation)
    
    return x


def __maxPool__(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')



def __ConvStack0__(x, channels_out, activation):
    with tf.variable_scope('layer1'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)
        
    return x


def __ConvStack1__(x, channels_out, activation):
    with tf.variable_scope('layer1'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)

    with tf.variable_scope('layer2'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)
        
    return x

 
def __ConvStack2__(x, channels_out, activation):
    with tf.variable_scope('layer1'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)

    with tf.variable_scope('layer2'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)
        
    with tf.variable_scope('layer3'):
        x = __Conv2D__(x,channels_out, filter_dim = 3, activation = activation)
        
    return x


def __Flatten__(x):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    with tf.variable_scope('flatten'):
        x = tf.reshape(x,[-1,dim])
    return x
       

               
def CPNet(x):

    with tf.variable_scope('down1') as scope:
        conv_1 = __ConvStack1__(x, channels_out = 64, activation = 'leaky_relu')
        pool_1 = __maxPool__(conv_1)

    with tf.variable_scope('down2') as scope:
        conv_2 = __ConvStack1__(pool_1, channels_out = 128, activation = 'leaky_relu')
        pool_2 = __maxPool__(conv_2)

    with tf.variable_scope('down3') as scope:
        conv_3 = __ConvStack1__(pool_2, channels_out = 256, activation = 'leaky_relu')
        pool_3 = __maxPool__(conv_3)

    with tf.variable_scope('flatten') as scope:
        flat_1 = __Flatten__(pool_3)
                  
    with tf.variable_scope('dense1') as scope:
        dense_1 = __Dense__(flat_1,12, activation = 'sigmoid')

    return dense_1
