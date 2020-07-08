# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:24:22 2020

@author: ASUS
"""
import sys
sys.path.append("./code")
sys.path.append("./breast_cancer_classifier-master")

import os
import tensorflow as tf
import tensorflow.keras as k
import datetime
from sklearn.metrics import roc_auc_score
from Augmentation import Feeding_Sequence
from All_keras_models import view_wise_model, breast_wise_model, joint_model, image_wise_model
from tensorflow.keras.utils import plot_model # pip install pydot-ng, pydot, graphviz
import numpy as np
import sklearn


class AUC_Malign(tf.keras.metrics.Metric):
    def __init__(self, name="AUC_Malign", **kwargs):# label = LM, RM, LB, RB
        super(AUC_Malign, self).__init__(name=name, **kwargs) # [0.6,0.75, 0.8]
        self.auc = tf.keras.metrics.AUC(name=name, thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    def update_state(self, y_true, y_pred):
        for i in range(4):
            if i%2 ==0:
                
                if i==0:
                    pred = y_pred[:,i]
                    true = y_true[:,i]
                else:
                    pred = tf.concat([pred, y_pred[:,i]], axis=-1)
                    true = tf.concat([true, y_true[:,i]], axis=-1)

        _ = self.auc.update_state(true, pred)

    def result(self):
        return self.auc.result()
        

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.auc.reset_states()



class AUC_Benign(tf.keras.metrics.Metric):
    def __init__(self, name="AUC_Benign", **kwargs):# label = LM, RM, LB, RB
        super(AUC_Benign, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC( name=name, thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    def update_state(self, y_true, y_pred):
        for i in range(4):
            if i%2!=0:
                if i==1:
                    pred = y_pred[:,i]
                    true = y_true[:,i]
                else:
                    pred = tf.concat([pred, y_pred[:,i]], axis=-1)
                    true = tf.concat([true, y_true[:,i]], axis=-1)

        _ = self.auc.update_state(true, pred)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.auc.reset_states()





class CustomCross(tf.keras.losses.Loss):
    def __init__(self, name='loss', **kwargs):
        super(CustomCross, self).__init__(name=name, **kwargs)
        
        self.num_classes = 4
        self.losses = []
        self.cross = tf.keras.losses.SparseCategoricalCrossentropy()


    def call(self, y_true, y_pred):
        for i in range(self.num_classes):
            pred = y_pred[:,i]
            true = y_true[:,i]

            diff = tf.ones(shape=tf.shape(pred), dtype = 'float32') - pred 
            true = tf.cast(tf.math.logical_not(tf.cast(true, dtype=tf.bool)), dtype ='float32') # cast from float to bool, negate and convert back to float
            pred = tf.stack([pred, diff], axis=-1)
            loss = self.cross(true, pred)

            
            if i==0:
                self.losses = tf.expand_dims(loss, -1)
            else:
                self.losses = tf.concat([self.losses, tf.expand_dims(loss, -1)], axis=-1)

        average_loss = tf.reduce_mean(self.losses)


        return average_loss




cropped_center_exam_list_path = './Data/INbreast_dataset/AllDICOMs_cropped/cropped_center_exam_list.pkl'
images_path = './Data/INbreast_dataset/AllDICOMs_cropped' 
xls_path = './Data/INbreast_dataset/INbreast.xls'   
heatmaps_path = './Data/INbreast_dataset/AllDICOMs_heatmaps'


# PARAMETERS

use_hdf5 = False
use_heatmaps = True
heatmaps_path = heatmaps_path
use_augmentation = True
seed = 1
num_epochs = 20
batch_size = 2
learning_rate = pow(10, -5)
training = True
validation_split = 0.1

print('\nUsing heatmaps:', use_heatmaps)

parameters = {
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": images_path,
        "batch_size": batch_size,
        "seed": seed,
        "augmentation": use_augmentation,
        "use_heatmaps": use_heatmaps,
        "heatmaps_path": heatmaps_path,
        "use_hdf5": use_hdf5,
        "training": training,
        "validation_split": validation_split
    }  



# PREPROCESSING DATA

feeding = Feeding_Sequence(x_set = cropped_center_exam_list_path,
                           y_set=xls_path,
                           parameters= parameters)

print('\nData loaded from', cropped_center_exam_list_path)


# MODEL AND MODEL PARAMETERS
k.backend.clear_session()
training = True


print('\nModel creation...')
#nome_modello = "image_wise_model"
#nome_modello = "joint_model"
#nome_modello = "view_wise_model"
nome_modello = "breast_wise_model"

if nome_modello == "joint_model":
	model = joint_model(training, use_heatmaps)
elif nome_modello == "image_wise_model":
    model = image_wise_model(training, use_heatmaps)
elif nome_modello == "breast_wise_model":
	model = breast_wise_model(training, use_heatmaps)
elif nome_modello == "view_wise_model":
	model = view_wise_model(training, use_heatmaps)
else:
	print("The name of the selected model is not valid.")

print('\n\nModel selected:', model.name)

#------------CREATE FOLDER FOR EACH MODEL WHEN NEEDED---------------#
if parameters['use_heatmaps']:
    model_dir = os.path.join('models', 'image_and_heatmaps', model.name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    model_dir = os.path.join('models', 'image_only', model.name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# ----- CALLBACKS ----- #	
tensorboard_dir = os.path.join(model_dir, "tensorboard")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=tensorboard_dir)

Early_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

checkpoint_path = os.path.join(model_dir, 'checkpoints', "cp.E{epoch:02d}_L{loss:.2f}_AUCM{AUC_Malign:.2f}_AUCB{AUC_Benign:.2f}.ckpt")
cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', save_weights_only =True, verbose=0, save_freq = 'epoch')

# --------------- COMPILE MODEL -----------#
loss=CustomCross()

model.compile(
    loss = loss,
    optimizer = k.optimizers.Adam(learning_rate=learning_rate),
    metrics = ['accuracy',
               AUC_Malign(),
               AUC_Benign()
               ]
    )


plot_model( model, to_file=os.path.join(model_dir, nome_modello + '.png'), show_shapes=True) # plot model

# ----- TRAINING ----- #
if training:
    print('\nTraining on GPU:', bool(len(tf.config.list_physical_devices('GPU'))))
    print("\nStarting training...")
    
    model.fit(     
        feeding,
     	epochs=num_epochs,
     	verbose = 1,
        workers=4,
        callbacks=[tensorboard_callback, Early_callback, cp_callback]
        )
    
    print("Training complete.")




# ----- SAVE MODEL ----- #
model.save(os.path.join(model_dir, model.name+".h5"))
print("\nModel saved at:", os.path.join(model_dir, model.name+".h5"))

