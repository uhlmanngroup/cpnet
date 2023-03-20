from model import CPNet, initialize_uninitialized
from loss import splineLoss, splinegen2d
import tensorflow as tf
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt



def predict(xdata,ydata):
    results = []
    results_control_pts = []
    results_loss = []

    with tf.Graph().as_default() as g:        
            
        dataset = dataset = tf.data.Dataset.from_tensor_slices((xdata,ydata)).batch(1)
        iterator = dataset.make_one_shot_iterator()
        x,y = iterator.get_next()
            
        with tf.variable_scope('model') as scope:
            control_pts = CPNet(x)
            contour = splinegen2d(control_pts)
        loss = tf.reduce_mean(splineLoss(y,control_pts))  
        saver = tf.train.Saver()
        
        with tf.Session() as sess:           
            saver.restore(sess, './checkpoints/model_trained.chpt')
            initialize_uninitialized(sess)    
            while True:
                try:
                    temp1,temp2,temp3 = sess.run([control_pts, contour, loss])
                    results_control_pts.append(temp1)
                    results.append(temp2)
                    results_loss.append(temp3)
                except tf.errors.OutOfRangeError:
                    break
            
    results = np.stack(results)
    results = np.squeeze(results)
    results_control_pts = np.stack(results_control_pts)
    results_control_pts = np.squeeze(results_control_pts)
    results_loss = np.stack(results_loss)
    results_loss = np.squeeze(results_loss)
    return (results, results_control_pts, results_loss)



x_test = np.load('./imgs_test.npy')
y_test = np.load('./contours_test.npy')

contours,control_pts, loss = predict(x_test,y_test)


dice = []

good = []
for ind in range(0,len(x_test)):

    img = x_test[ind]
    #img = rescale(img)
    img = np.squeeze(img)
    img [img>0] = 1

    #change here
    
    X = 64*contours[ind,:,0] 
    Y = 64*contours[ind,:,1]

    img_ = np.zeros((64,64))
    X = np.rint(X).astype(int)
    Y = np.rint(Y).astype(int)

    for i in range(len(X)):
        img_[X[i],Y[i]] = 1

    img_ = np.transpose(img_)
    img_ = ndimage.binary_fill_holes(img_)


    intersection = np.logical_and(img, img_)
    union = np.logical_or(img, img_)
    dice = np.append(dice, ((2*intersection.sum())/(img.sum()+img_.sum()))) 
    
print('median_dice = ', np.median(dice))




#for ind in range(x_test.shape[0]):
# for ind in range(0,20):
#     X = 64*contours[ind,:,0]
#     Y = 64*contours[ind,:,1]
    
#     #X = np.rint(X).astype(int)
#     #Y = np.rint(Y).astype(int)

#     X1 = 64*control_pts[ind][0:6]
#     Y1 = 64*control_pts[ind][6:12]
    
#     X2 = 64*y_test[ind,:,0]
#     Y2 = 64*y_test[ind,:,1]
    
#     img = np.tile(x_test[ind], (1,1,3))
  
#     plt.figure()
#     plt.imshow(img)
#     plt.plot(X,Y, 'r')
#     #plt.plot(X1,Y1, 'bo')
#     #plt.plot(X2,Y2, 'g')
#     plt.title('Loss' + str(ind) + '=' + str(loss[ind]))
#     plt.show()