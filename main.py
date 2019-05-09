from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from misc import *
from inception_network import *
np.set_printoptions(threshold=np.nan)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha=0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):

    # Step 1: Compute the encoding for the image.
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding - database[identity])

    if dist < 0.7:
        print("It's " + str(identity) + ", Face Recognized")
    else:
        print("It's not " + str(identity) + ", Not Recognized")


    return dist

verify("images/camera_0.jpg", "younes", database, FRmodel)

verify("images/camera_2.jpg", "kian", database, FRmodel)


def who_is_it(image_path, database, model):

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding
    min_dist = 108

    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name


    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("It's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

who_is_it("images/camera_2.jpg", database, FRmodel)
