from model import CPNet, initialize_uninitialized
from loss import splineLoss, splinegen2d
import tensorflow as tf
import numpy as np
import math
import sys
import matplotlib.pyplot as plt


def train(xdata_train, ydata_train, xdata_val, ydata_val, num_epochs = 10, batch_size = 32, lr = 0.001):
    num_train = xdata_train.shape[0]     
    num_val = xdata_val.shape[0]

    with tf.Graph().as_default() as g:
        bs = tf.placeholder(tf.int64)
        train_dataset = tf.data.Dataset.from_tensor_slices((xdata_train,ydata_train)).shuffle(num_train).batch(bs)            
        val_dataset = tf.data.Dataset.from_tensor_slices((xdata_val,ydata_val)).batch(bs)           
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        x,y = iterator.get_next()
        train_init_op = iterator.make_initializer(train_dataset)
        val_init_op = iterator.make_initializer(val_dataset)
        
        prediction = CPNet(x)
        
        loss = tf.reduce_mean(splineLoss(y,prediction))
        global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.exponential_decay(lr,global_step,3000, 0.96, staircase=True)
        train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
        saver = tf.train.Saver()

        with tf.Session() as sess:           
            sess.run(tf.global_variables_initializer())
                
            saveweights = 999999
            epoch_length = []
            loss_train = []
            loss_val = []            
            
            for epoch in range(num_epochs):
                #print(sess.run(lr))
                sess.run(train_init_op, feed_dict={bs: batch_size})
                num_batches = 0
                to_print = ''
                while True:
                    try:
                        _,batch_loss = sess.run([train,loss])
                        num_batches += 1
                        percentage_done = int(100*num_batches*batch_size/float(num_train))
                        bar = int(percentage_done/5)
                        to_print = '\r' + 'Epoch ' + str(epoch) + ' |' + bar*'=' + '>' + (20-bar)*' ' + '|  ' + str(percentage_done) + ' % complete' + '| train loss = ' + str(batch_loss)
                        print(to_print, end = '\r')
                        sys.stdout.flush()
                    except tf.errors.OutOfRangeError:
                        print(len(to_print)*' ', end='\r')
                        break

                sess.run(val_init_op, feed_dict={bs: num_val})
                val_loss = sess.run(loss)                
                epoch_print = '\r' + 'Epoch ' + str(epoch) + ' |' + 20*'=' + '>| 100 % complete' + ' |'
                epoch_print += ' train loss = ' + str(batch_loss) + ' | val loss = ' + str(val_loss) + ' |'
                print(epoch_print)
                
                epoch_length.append(epoch)
                loss_train.append(batch_loss)
                loss_val.append(val_loss)
                
                
                if (saveweights >= val_loss):
                    saver.save(sess,'./checkpoints/model_best.chpt', write_meta_graph=False)
                    saveweights = val_loss
                
            plt.plot(epoch_length, loss_train, 'b', label = "Training Loss")
            plt.plot(epoch_length, loss_val, 'r', label = "Validation Loss")    
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
#             plt.savefig('shallow1_5cp.png')
            plt.show()
                
            saver.save(sess,'./checkpoints/model.chpt', write_meta_graph=False)
            
    return (epoch_length, loss_train, loss_val)



x_train = np.load('./imgs_train_.npy')
y_train = np.load('./contours_train_.npy')

x_val = np.load('./imgs_test_.npy')
y_val = np.load('./contours_test_.npy')


epoch_length, loss_train, loss_val = train(x_train,y_train,x_val,y_val,100,32,0.0001)