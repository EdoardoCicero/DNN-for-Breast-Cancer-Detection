# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:33:21 2020

@author: Edoardo
"""
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append("../breast_cancer_classifier-master/")

import src.utilities.pickling as pickling
import src.data_loading.loading as loading
from src.constants import VIEWS #, VIEWANGLES, LABELS, MODELMODES
import math
from tensorflow.keras.utils import Sequence
from creaListeDiImmaginiElabelsPerView import *



def get_output_dict(xls_path, images_path): # get from xls with ground truth to dictionary of labels
    output_labels_path = xls_path # path to xls file
    
    # OUTPUT 
    screen2label = allScreen2label(output_labels_path, images_path)
    listaImg, listaLbl = suddividiPercorsiImmaginiElabelsInListe(images_path, screen2label) # list of images, list of labels
    patient_label_dict = {} # key = patient name, value = onehot label

    
    for n_pat in range(int(len(listaImg)/4)): #iterate for every patient
        patient = listaImg[n_pat*4].split('_')[1]
        patient_label_dict[patient] = []

        for n_img, image in enumerate(listaImg[n_pat*4:n_pat*4+4]): # iterate for every image associated to that patient
            patient_label_dict[patient].append(listaLbl[n_pat*4 + n_img]) # add onehot label to patient

        patient_label_dict[patient] = np.sum(np.array(patient_label_dict[patient]), axis=0).astype(bool).astype('uint8')
       
    output_dict = patient_label_dict
    
    return output_dict #dictionary of patients with associated output labels



# DATA SEQUENCE AS MODEL INPUT 
class Feeding_Sequence(Sequence):

    def __init__(self, x_set, y_set, parameters): # x_set = path to exam_list_cropped, y_set= path to xls file
        # INPUT
        val_split = parameters["validation_split"] # split rate
        train_split = 1 - val_split 
        total_patients = len(pickling.unpickle_from_file(x_set))
        number_patients_for_training = int((total_patients)*(train_split))

        if parameters["training"] == True: # if doing the validation split
            x_set = pickling.unpickle_from_file(x_set)[:number_patients_for_training] # list of dictionaries from exam_list_cropped.pkl
        else:
            x_set = pickling.unpickle_from_file(x_set)[number_patients_for_training:]
            
        y_set = get_output_dict(xls_path = y_set, images_path = parameters["image_path"]) # get output dict
        
        self.random_number_generator = np.random.RandomState(parameters["seed"])
        self.x, self.y = x_set, y_set
        self.batch_size = parameters['batch_size']
        self.parameters = parameters
        
        
         
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_dict = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        batch_y = []
        batch_x = []
        batch_x_feed = {}
        batch_x_feed_new = {}
        for view in VIEWS.LIST:
            batch_x_feed_new[view] = []
            
        test_x_var = []
        test_y_var = []
       
        image_extension = ".hdf5" if self.parameters["use_hdf5"] else ".png"
        image_index = 0

        for datum in (batch_x_dict): # THIS AUGMENTATION PART IS PARTIALLY TAKEN FROM run_model.py

            patient = datum['L-CC'][0].split('_')[1] # get name of patient
            patient_screens = []
            patient_screens_dict = {}
            

            patient_dict = {view: [] for view in VIEWS.LIST} # create structure that will be filled with images
            for view in VIEWS.LIST:
                patient_screens_dict[view] = []
                
                short_file_path = datum[view][image_index] # name of image file associated to that view
                test_x_var.append(short_file_path)
                
                loaded_image = loading.load_image(
                    image_path=os.path.join(self.parameters["image_path"], short_file_path + image_extension),
                    view=view,
                    horizontal_flip=datum["horizontal_flip"],
                )
                if self.parameters["use_heatmaps"]:
                    loaded_heatmaps = loading.load_heatmaps(
                        benign_heatmap_path=os.path.join(self.parameters["heatmaps_path"], "heatmap_benign",
                                                         short_file_path + ".hdf5"),
                        malignant_heatmap_path=os.path.join(self.parameters["heatmaps_path"], "heatmap_malignant",
                                                            short_file_path + ".hdf5"),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )
                else:
                    loaded_heatmaps = None


                if self.parameters["augmentation"]:
                    image_index = self.random_number_generator.randint(low=0, high=len(datum[view]))
                cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                    image=loaded_image,
                    auxiliary_image=loaded_heatmaps,
                    view=view,
                    best_center=datum["best_center"][view][image_index],
                    random_number_generator=self.random_number_generator,
                    augmentation=self.parameters["augmentation"],
                    max_crop_noise=self.parameters["max_crop_noise"],
                    max_crop_size_noise=self.parameters["max_crop_size_noise"],
                )
                if loaded_heatmaps is None:
                    patient_dict[view].append(cropped_image[:, :, np.newaxis])
                else:
                    patient_dict[view].append(np.concatenate([
                        cropped_image[:, :, np.newaxis],
                        cropped_heatmaps,
                    ], axis=2))


                batch_x_feed_new[view].append(patient_dict[view][-1]) # adding images of specific view to dictionary of patient

            batch_y.append(self.y[patient]) # output related to the patient
           

        batch_y = np.stack(np.array(batch_y), axis=0)
        
        # IF U WANT (BATCH, 4, 2) DECOMMENT THIS
        # for ni, i in enumerate(batch_y):
        #     if ni==0:
        #         #batch_y_tf = tf.expand_dims(tf.one_hot(i,2), 0)
        #         batch_y_tf = tf.expand_dims(tf.cast(tf.math.logical_not(tf.cast(tf.one_hot(i,2), dtype=tf.bool)), dtype='float32'), 0)
        #         #print(batch_y_tf)
        #     else:
        #         new_patient = tf.expand_dims(tf.cast(tf.math.logical_not(tf.cast(tf.one_hot(i,2), dtype=tf.bool)), dtype='float32'), 0)
        #         batch_y_tf = tf.concat([batch_y_tf, new_patient] , axis =0)
        # #print(batch_y_tf)
        # for n, view in enumerate(VIEWS.LIST):
        #     batch_x_feed[view] = np.moveaxis(np.stack(np.array(batch_x)[:, n], axis=0), -1,1) # dictionary: key=scan(i.e. 'L-CC', 'R-CC'...), value=tensor of scans (1 image per patient)
        
        for n, view in enumerate(VIEWS.LIST):
            batch_x_feed_new[view] = np.moveaxis(np.stack(np.array(batch_x_feed_new[view]), axis=0), -1,1) # to obtain (1, width, height)
            
        # returns AS X a dict with all images for all patient patient organized by keys
        # that are the views (i.e. {'L-CC: [all L-CC iamges of patients per batch], 'L-MLO': ...})
        # returns as y a tensor of [batch, classes]
        return batch_x_feed_new , batch_y#, [None] 
            

 


