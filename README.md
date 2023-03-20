# CPNet

A Learning Based Formulation of Parametric Curve Fitting for Bioimage Analysis.

model.py contains the network design.<br />
loss.py contains the loss function for training. <br />
Run training.py for training.<br />
Run test.py for testing. <br />

We provide trained weights for the network for quick testing on [BBBC038](https://data.broadinstitute.org/bbbc/BBBC038/) in the checkpoints folder. We also provide the [dataset](https://drive.google.com/file/d/1LEsgejb9XQcXagivhggbh1kgcKky_-F6/view?usp=sharing). For using over a new dataset, just replace the imgs_train.npy (training images), contours_train.npy (training labels), imgs_test.npy (test images), and contours_test.npy (test labels) with relevant files. 

Requirements - <br />
numpy <br />
Tensorflow 1.14