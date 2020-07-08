# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:24:28 2020

@author: Edoardo
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

class CustomCross(tf.keras.losses.Loss):
    def __init__(self, name="custom_cross"):
        super().__init__(name=name)
        self.num_classes = 4
        #self.losses = []
        #self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        for i in range(self.num_classes):
            pred = y_pred[:,i]
            print('pred_0:', pred)
            true = y_true[:,i]
            print('true:', true)
            #pred = tf.expand_dims(pred, axis=-1 )
            print(tf.expand_dims(pred, axis=-1 ).shape)
            diff = tf.math.subtract(tf.ones(shape=tf.shape(pred)) , pred )
            print('diff:', diff)
            pred = tf.stack([pred, diff], axis=-1)
            print('pred_1:', pred)

            loss = tf.keras.losses.sparse_categorical_crossentropy(true, pred)
            if i==0:
                self.losses = loss
            else:
                self.losses = tf.concat([self.losses, loss], axis=-1)
            print('self.losses:', self.losses)
        average_loss = tf.reduce_mean(self.losses)
        print(average_loss)
        return average_loss
    
    
    
class BinaryTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    #self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
      
      
    #sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_weights(sample_weight, values)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives





