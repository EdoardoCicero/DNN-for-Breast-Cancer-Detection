# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:19:18 2020

@author: Edoardo
"""
import tensorflow as tf
import tensorflow.keras as k
import sys
sys.path.append('../breast_cancer_classifier-master')
import src.modeling.models_keras_2 as models
import math



# input_cc_shape = (1, 2677, 1942)
# input_mlo_shape = (1, 2974, 1748)



def r22cc ( x, training ):
   with tf.compat.v1.variable_scope("r22cc"):
       return models.resnet22(x,training)
    

def r22mlo ( x, training ):
    with tf.compat.v1.variable_scope("r22mlo"):
        return models.resnet22(x,training)
    
    
    
    

def breast_wise_model(training, use_heatmaps):
    
    dim_0 = int(use_heatmaps)* 2 + 1

    input_cc_shape = (dim_0, 2677, 1942)
    input_mlo_shape = (dim_0, 2974, 1748)
    
   
    # ---------- R22 MODELS ---------- #
    CC_R22_input = k.Input(shape=input_cc_shape, name="input_R22_CC")
    CC_R22_output = r22cc(CC_R22_input, training)
    CC_R22_model = k.Model(inputs= CC_R22_input, outputs= CC_R22_output, name='CC_R22_model' )

    MLO_R22_input = k.Input(shape=input_mlo_shape, name="input_R22_MLO")
    MLO_R22_output = r22mlo(MLO_R22_input, training)
    MLO_R22_model = k.Model(inputs= MLO_R22_input, outputs= MLO_R22_output, name = 'MLO_R22_model')

    # ------- MODEL INPUTS ----------- #
    inp_RCC = k.Input(shape = input_cc_shape, name = 'R-CC')
    inp_LCC = k.Input(shape = input_cc_shape, name = 'L-CC')
    inp_RMLO = k.Input(shape = input_mlo_shape, name = 'R-MLO')
    inp_LMLO = k.Input(shape = input_mlo_shape, name = 'L-MLO')
    
    R_CC = CC_R22_model(inp_RCC)
    L_CC = CC_R22_model(inp_LCC)
    R_MLO = MLO_R22_model(inp_RMLO)
    L_MLO = MLO_R22_model(inp_LMLO)
    
    avg_pool_LCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "lccAveragePooling")(L_CC))
    avg_pool_RCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "rccAveragePooling")(R_CC))
    
    avg_pool_LMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "lmloAveragePooling")(L_MLO))
    avg_pool_RMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "rmloAveragePooling")(R_MLO))
   
    
    # ---------- CONCAT AND BEYOND ------------ #
    
    L_concat = k.layers.concatenate([avg_pool_LCC, avg_pool_LMLO], axis=1)
    R_concat = k.layers.concatenate([avg_pool_RCC, avg_pool_RMLO], axis=1)
    
    L_relu = k.layers.Dense(512, name='L_relu', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(L_concat)
    R_relu = k.layers.Dense(512, name='R_relu', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(R_concat)
    
    
    L_log_0 = k.layers.Dense(2,  activation = 'softmax', name= 'L_fully_0',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(L_relu)
    L_log_1 = k.layers.Dense(2, activation = 'softmax', name= 'L_fully_1',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(L_relu)
    L_stack = tf.stack([L_log_0, L_log_1],axis=1, name = 'L_stack')
    
    R_log_0 = k.layers.Dense(2, activation = 'softmax', name= 'R_fully_0',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(R_relu)
    R_log_1 = k.layers.Dense(2, activation = 'softmax', name= 'R_fully_1',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(R_relu)
    R_stack = tf.stack([ R_log_0, R_log_1],axis=1, name = 'R_stack')
    
    
    # ----- OUTPUT OF FULL MODEL ----------- #
    
    concat = k.layers.concatenate([L_stack, R_stack],axis=1, name = 'concat')[:,:,0]
    #concat = k.layers.Flatten()(tf.slice(concat, [0,0,0], [-1,-1, 1]))

    # model_output = tf.math.exp( concat, name = 'exp_output' )
    model_output = concat
    
    model = k.Model(inputs=[inp_LMLO, inp_RMLO, inp_LCC, inp_RCC], outputs = model_output, name='breast_wise_model')
    model.summary()
    
    
    return model


#breast_wise_model(True)



def view_wise_model(training, use_heatmaps):
    
    dim_0 = int(use_heatmaps)* 2 + 1

    input_cc_shape = (dim_0, 2677, 1942)
    input_mlo_shape = (dim_0, 2974, 1748)
    

    # ---------- R22 MODELS ---------#
    CC_R22_input = k.Input(shape=input_cc_shape, name="input_R22_CC")
    CC_R22_output = r22cc(CC_R22_input, training)
    CC_R22_model = k.Model(inputs= CC_R22_input, outputs= CC_R22_output, name='CC_R22_model' )

    
    MLO_R22_input = k.Input(shape=input_mlo_shape, name="input_R22_MLO")
    MLO_R22_output = r22mlo(MLO_R22_input, training)
    MLO_R22_model = k.Model(inputs= MLO_R22_input, outputs= MLO_R22_output, name = 'MLO_R22_model')


    # ---------- CC BRANCH -----------#
    inp_RCC = k.Input(shape = input_cc_shape, name = 'R-CC')
    inp_LCC = k.Input(shape = input_cc_shape, name = 'L-CC')
    R_CC = CC_R22_model(inp_RCC)
    L_CC = CC_R22_model(inp_LCC)
    
    avg_pool_LCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "lccAveragePooling")(L_CC))
    avg_pool_RCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "rccAveragePooling")(R_CC))
    
    CC_concat = k.layers.concatenate([avg_pool_LCC, avg_pool_RCC], axis=1)
    
    relu_CC = k.layers.Dense(512, name='relu_CC', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(CC_concat)

    
    log_CC_0 = tf.nn.log_softmax(k.layers.Dense(2, name='log_CC_0')(relu_CC), axis= -1)
    log_CC_1 = tf.nn.log_softmax(k.layers.Dense(2, name='log_CC_1')(relu_CC), axis= -1)
    log_CC_2 = tf.nn.log_softmax(k.layers.Dense(2, name='log_CC_2')(relu_CC), axis= -1)
    log_CC_3 = tf.nn.log_softmax(k.layers.Dense(2, name='log_CC_3')(relu_CC), axis= -1)

    CC_stack = tf.stack([log_CC_0, log_CC_1, log_CC_2, log_CC_3],axis=1, name = 'CC_stack')

    
    
    # ---------- MLO BRANCH ------------ #
    
    inp_RMLO = k.Input(shape = input_mlo_shape, name = 'R-MLO')
    inp_LMLO = k.Input(shape = input_mlo_shape, name = 'L-MLO')
    R_MLO = MLO_R22_model(inp_RMLO)
    L_MLO = MLO_R22_model(inp_LMLO)
    avg_pool_LMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "lmloAveragePooling")(L_MLO))
    avg_pool_RMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "rmloAveragePooling")(R_MLO))
    
    MLO_concat = k.layers.concatenate([avg_pool_LMLO, avg_pool_RMLO], axis=1)
    
    relu_MLO = k.layers.Dense(512, name='relu_MLO', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(MLO_concat)

    
    log_MLO_0 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_MLO), axis = -1, name='log_MLO_0')
    log_MLO_1 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_MLO), axis = -1, name='log_MLO_1')
    log_MLO_2 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_MLO), axis = -1, name='log_MLO_2')
    log_MLO_3 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_MLO), axis = -1, name='log_MLO_3')

    MLO_stack = tf.stack([log_MLO_0, log_MLO_1, log_MLO_2, log_MLO_3],axis=1, name = 'MLO_stack')

    
    
    # ---------- OUTPUT OF MODEL ---------- #
    
    average = k.layers.Average(name="average")([CC_stack, MLO_stack])[:,:,0]
    model_output = tf.math.exp( average, name = 'exp_output' )

    model = k.Model(inputs=[inp_LMLO, inp_RMLO, inp_LCC, inp_RCC], outputs = model_output, name='view_wise_model')
    model.summary()
    
    
    return model
    

#view_wise_model(True)







def joint_model(training, use_heatmaps):
    
    dim_0 = int(use_heatmaps)* 2 + 1

    input_cc_shape = (dim_0, 2677, 1942)
    input_mlo_shape = (dim_0, 2974, 1748)
    
    # ---------- R22 MODELS ---------#
    CC_R22_input = k.Input(shape=input_cc_shape, name="input_R22_CC")
    CC_R22_output = r22cc(CC_R22_input, training)
    CC_R22_model = k.Model(inputs= CC_R22_input, outputs= CC_R22_output, name='CC_R22_model' )

    
    MLO_R22_input = k.Input(shape=input_mlo_shape, name="input_R22_MLO")
    MLO_R22_output = r22mlo(MLO_R22_input, training)
    MLO_R22_model = k.Model(inputs= MLO_R22_input, outputs= MLO_R22_output, name = 'MLO_R22_model')


    # ---------- CC BRANCH -----------#
    inp_RCC = k.Input(shape = input_cc_shape, name = 'R-CC')
    inp_LCC = k.Input(shape = input_cc_shape, name = 'L-CC')
    R_CC = CC_R22_model(inp_RCC)
    L_CC = CC_R22_model(inp_LCC)
    
    avg_pool_LCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "lccAveragePooling")(L_CC))
    avg_pool_RCC = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "rccAveragePooling")(R_CC))

    # ---------- MLO BRANCH ------------ #
    
    inp_RMLO = k.Input(shape = input_mlo_shape, name = 'R-MLO')
    inp_LMLO = k.Input(shape = input_mlo_shape, name = 'L-MLO')
    R_MLO = MLO_R22_model(inp_RMLO)
    L_MLO = MLO_R22_model(inp_LMLO)
    avg_pool_LMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "lmloAveragePooling")(L_MLO))
    avg_pool_RMLO = k.layers.Flatten()(k.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "rmloAveragePooling")(R_MLO))

        
    concatenation = k.layers.concatenate([avg_pool_LCC, avg_pool_RCC, avg_pool_LMLO, avg_pool_RMLO], axis=1)
    relu = k.layers.Dense(512, name='relu', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(concatenation)
    
    
    log_0 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu), axis = -1, name='log_0')
    log_1 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu), axis = -1, name='log_1')
    log_2 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu), axis = -1, name='log_2')
    log_3 = tf.nn.log_softmax(k.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu), axis = -1, name='log_3')

    stack = tf.stack([log_0, log_1, log_2, log_3], axis=1)[:,:,0]
    model_output = tf.math.exp( stack, name = 'exp_output' )

    model = k.Model(inputs=[inp_LMLO, inp_RMLO, inp_LCC, inp_RCC], outputs = model_output, name='joint_model')
    model.summary()

    
    return model


#joint_model(True)
	
def image_wise_model(training, use_heatmaps):
    
    dim_0 = int(use_heatmaps)* 2 + 1

    input_cc_shape = (dim_0, 2677, 1942)
    input_mlo_shape = (dim_0, 2974, 1748)	
    # ---------- R22 MODELS ---------#
    CC_R22_input = tf.keras.Input(shape=input_cc_shape, name="input_R22_CC")
    CC_R22_output = r22cc(CC_R22_input, training)
    CC_R22_model = tf.keras.Model(inputs= CC_R22_input, outputs= CC_R22_output)

    MLO_R22_input = tf.keras.Input(shape=input_mlo_shape, name="input_R22_MLO")
    MLO_R22_output = r22mlo(MLO_R22_input, training)
    MLO_R22_model = tf.keras.Model(inputs= MLO_R22_input, outputs= MLO_R22_output)
	

    # ---------- CC BRANCH -----------#
	
    inp_RCC = tf.keras.Input(shape = input_cc_shape, name = 'R-CC')
    inp_LCC = tf.keras.Input(shape = input_cc_shape, name = 'L-CC')
    R_CC = CC_R22_model(inp_RCC)
    L_CC = CC_R22_model(inp_LCC)
	
    avg_pool_LCC = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "lccAveragePooling")(L_CC))
    avg_pool_RCC = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling2D(data_format = "channels_first", pool_size=(42,31), name = "rccAveragePooling")(R_CC))

    relu_LCC = tf.keras.layers.Dense(512, name='relu_LCC', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(avg_pool_LCC)
    relu_RCC = tf.keras.layers.Dense(512, name='relu_RCC', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(avg_pool_RCC)
	
    log_LCC_0 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_LCC), name='log_LCC_0', axis = -1)
    log_LCC_1 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_LCC), name='log_LCC_1', axis = -1)
	
    log_RCC_0 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_RCC), name='log_RCC_0', axis = -1)
    log_RCC_1 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_RCC), name='log_RCC_1', axis = -1)
	
    LCC_stack = tf.stack([log_LCC_0, log_LCC_1], name = "LCC_stack", axis = 1)
    RCC_stack = tf.stack([log_RCC_0, log_RCC_1], name = "RCC_stack", axis = 1)


    # ---------- MLO BRANCH ------------ #
	
    inp_RMLO = tf.keras.Input(shape = input_mlo_shape, name = 'R-MLO')
    inp_LMLO = tf.keras.Input(shape = input_mlo_shape, name = 'L-MLO')
    R_MLO = MLO_R22_model(inp_RMLO)
    L_MLO = MLO_R22_model(inp_LMLO)
   	
    avg_pool_LMLO = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "lmloAveragePooling")(L_MLO))
    avg_pool_RMLO = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling2D(data_format = "channels_first", pool_size=(47,28), name = "rmloAveragePooling")(R_MLO))
   	
    relu_LMLO = tf.keras.layers.Dense(512, name='relu_LMLO', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(avg_pool_LMLO)
    relu_RMLO = tf.keras.layers.Dense(512, name='relu_RMLO', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(avg_pool_RMLO)
   
    log_LMLO_0 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_LMLO), name='log_LMLO_0',  axis = -1)
    log_LMLO_1 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_LMLO), name='log_LMLO_1', axis = -1)
   	
    log_RMLO_0 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_RMLO), name='log_RMLO_0', axis = -1)
    log_RMLO_1 = tf.nn.log_softmax(tf.keras.layers.Dense(2,kernel_regularizer = tf.keras.regularizers.l2(l=math.pow(10,-4.5)))(relu_RMLO), name='log_RMLO_1', axis = -1)	
   	
    LMLO_stack = tf.stack([log_LMLO_0, log_LMLO_1], name = "LMLO_stack", axis = 1)
    RMLO_stack = tf.stack([log_RMLO_0, log_RMLO_1], name = "RMLO_stack", axis = 1)
   
   	
   	#----- CONCATENATION AND BEYOND ------ #
   
    CC_concatenation = tf.keras.layers.concatenate([LCC_stack, RCC_stack], axis = 1, name = "CC_concatenation")
    MLO_concatenation = tf.keras.layers.concatenate([LMLO_stack, RMLO_stack], axis = 1, name = "MLO_concatenation")
   	
    average = tf.keras.layers.Average(name="average")([CC_concatenation,MLO_concatenation])[:,:,0]


# ----- OUTPUT OF FULL MODEL ----------- #

    model_output = tf.math.exp(average, name = 'exp_output')
	
    model = tf.keras.Model(inputs=[inp_LMLO, inp_RMLO, inp_LCC, inp_RCC], outputs = model_output, name = "image_wise_model")
    model.summary()
	
    return model