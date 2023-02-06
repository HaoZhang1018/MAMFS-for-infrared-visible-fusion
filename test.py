# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import scipy.io as scio


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(fea_IR,fea_VI, reuse,update_collection=None):
    with tf.variable_scope('fusion_model',reuse=reuse): 
        input_fea_cat = tf.concat([fea_IR,fea_VI],axis=-1)    
######################################################### 
#################### Layer 1 ############################     
        with tf.variable_scope('Fusion_layer1'):
            weights=tf.get_variable("Fusion_w1",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer1/Fusion_w1')))
            bias=tf.get_variable("Fusion_b1",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer1/Fusion_b1')))
            Fusion_conv1=tf.nn.conv2d(input_fea_cat, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv1 = lrelu(Fusion_conv1) 
######################################################### 
#################### Layer 2 ############################      
        with tf.variable_scope('Fusion_layer2'):
            weights=tf.get_variable("Fusion_w2",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer2/Fusion_w2')))
            bias=tf.get_variable("Fusion_b2",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer2/Fusion_b2')))
            Fusion_conv2=tf.nn.conv2d(Fusion_conv1, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv2 = lrelu(Fusion_conv2)       
######################################################### 
#################### Layer 3 ############################       
        with tf.variable_scope('Fusion_layer3'):
            weights=tf.get_variable("Fusion_w3",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer3/Fusion_w3')))
            bias=tf.get_variable("Fusion_b3",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer3/Fusion_b3')))
            Fusion_conv3=tf.nn.conv2d(Fusion_conv2, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv3 = lrelu(Fusion_conv3)  
            
######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_ir'):
            weights=tf.get_variable("Fusion_w4_ir",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_ir/Fusion_w4_ir')))
            bias=tf.get_variable("Fusion_b4_ir",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_ir/Fusion_b4_ir')))
            Fusion_conv4_ir =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_ir = lrelu(Fusion_conv4_ir)

######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_vi'):
            weights=tf.get_variable("Fusion_w4_vi",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_vi/Fusion_w4_vi')))
            bias=tf.get_variable("Fusion_b4_vi",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_vi/Fusion_b4_vi')))
            Fusion_conv4_vi =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_vi = lrelu(Fusion_conv4_vi)
            
######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_bias'):
            weights=tf.get_variable("Fusion_w4_bias",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_bias/Fusion_w4_bias')))
            bias=tf.get_variable("Fusion_b4_bias",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer4_bias/Fusion_b4_bias')))
            Fusion_conv4_bias =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_bias = lrelu(Fusion_conv4_bias)
            
######################################################### 
#################### Layer 5 #########################            
        with tf.variable_scope('Fusion_layer5_ir'):
            weights=tf.get_variable("Fusion_w5_ir",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_ir/Fusion_w5_ir')))
            bias=tf.get_variable("Fusion_b5_ir",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_ir/Fusion_b5_ir')))
            Fusion_conv5_ir =tf.nn.conv2d(Fusion_conv4_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_ir = tf.nn.sigmoid(Fusion_conv5_ir)           

######################################################### 
#################### Layer 6 #########################            
        with tf.variable_scope('Fusion_layer5_vi'):
            weights=tf.get_variable("Fusion_w5_vi",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_vi/Fusion_w5_vi')))
            bias=tf.get_variable("Fusion_b5_vi",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_vi/Fusion_b5_vi')))
            Fusion_conv5_vi =tf.nn.conv2d(Fusion_conv4_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_vi = tf.nn.sigmoid(Fusion_conv5_vi)  

######################################################### 
#################### Layer 6 #########################            
        with tf.variable_scope('Fusion_layer5_bias'):
            weights=tf.get_variable("Fusion_w5_bias",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_bias/Fusion_w5_bias')))
            bias=tf.get_variable("Fusion_b5_bias",initializer=tf.constant(reader_Fusion.get_tensor('fusion_model/Fusion_layer5_bias/Fusion_b5_bias')))
            Fusion_conv5_bias =tf.nn.conv2d(Fusion_conv4_bias, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_bias = lrelu(Fusion_conv5_bias)  

                                                                                                        
    return Fusion_conv5_ir, Fusion_conv5_vi,Fusion_conv5_bias


def Extract_model(img,reuse,update_collection=None):
    with tf.variable_scope('Extract_model',reuse=reuse):   
#########################################################     
####################   Encoder   ########################     
#########################################################     
    
######################################################### 
###################### Layer 1 ##########################      
        with tf.variable_scope('EN_layer1'):
            weights=tf.get_variable("EN_w1",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer1/EN_w1')))
            bias=tf.get_variable("EN_b1",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer1/EN_b1')))
            EN_conv1 = tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv1 = lrelu(EN_conv1)

####################### Block 1 ######################### 
####################### Layer 2 #########################   
        with tf.variable_scope('EN_layer2'):
            weights=tf.get_variable("EN_w2",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer2/EN_w2')))
            bias=tf.get_variable("EN_b2",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer2/EN_b2')))
            EN_conv2=tf.nn.conv2d(EN_conv1, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv2 = lrelu(EN_conv2)   
####################### Layer 3 #########################   
        with tf.variable_scope('EN_layer3'):
            weights=tf.get_variable("EN_w3",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer3/EN_w3')))
            bias=tf.get_variable("EN_b3",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer3/EN_b3')))
            EN_conv3=tf.nn.conv2d(EN_conv2, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv3 = lrelu(EN_conv3) 
                        
        dense_cat_23=tf.concat([EN_conv2,EN_conv3],axis=-1)             
            
####################### Layer 4 #########################   
        with tf.variable_scope('EN_layer4'):
            weights=tf.get_variable("EN_w4",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer4/EN_w4')))
            bias=tf.get_variable("EN_b4",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer4/EN_b4')))
            EN_conv4=tf.nn.conv2d(dense_cat_23, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv4 = lrelu(EN_conv4)            
            
        dense_cat_234=tf.concat([EN_conv2,EN_conv3,EN_conv4],axis=-1)             
            
####################### Layer 5 #########################   
        with tf.variable_scope('EN_layer5'):
            weights=tf.get_variable("EN_w5",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer5/EN_w5')))
            bias=tf.get_variable("EN_b5",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer5/EN_b5')))
            EN_conv5=tf.nn.conv2d(dense_cat_234, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv5 = lrelu(EN_conv5)             
#################  Spatial Attention 1 ####################   
        SAttent_1_max=tf.reduce_max(EN_conv5, axis=3, keepdims=True)
        SAttent_1_mean=tf.reduce_mean(EN_conv5, axis=3, keepdims=True)
        SAttent_1_cat_mean_max=tf.concat([SAttent_1_max,SAttent_1_mean],axis=-1)        
        with tf.variable_scope('layer1_atten_map'):
            weights=tf.get_variable("w1_atten_map",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_atten_map/w1_atten_map')))
            bias=tf.get_variable("b1_atten_map",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_atten_map/b1_atten_map')))
            conv1_atten_map=tf.nn.conv2d(SAttent_1_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_atten_map = tf.nn.sigmoid(conv1_atten_map)  
        block_1_atten_out= EN_conv5*conv1_atten_map           
            
        block_12_out = lrelu(block_1_atten_out+EN_conv1)           
            
            
####################### Block 2 ######################### 
####################### Layer 6 #########################   
        with tf.variable_scope('EN_layer6'):
            weights=tf.get_variable("EN_w6",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer6/EN_w6')))
            bias=tf.get_variable("EN_b6",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer6/EN_b6')))
            EN_conv6=tf.nn.conv2d(block_12_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv6 = lrelu(EN_conv6)   
####################### Layer 7 #########################   
        with tf.variable_scope('EN_layer7'):
            weights=tf.get_variable("EN_w7",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer7/EN_w7')))
            bias=tf.get_variable("EN_b7",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer7/EN_b7')))
            EN_conv7=tf.nn.conv2d(EN_conv6, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv7 = lrelu(EN_conv7) 
                        
        dense_cat_67=tf.concat([EN_conv6,EN_conv7],axis=-1)             
            
####################### Layer 8 #########################   
        with tf.variable_scope('EN_layer8'):
            weights=tf.get_variable("EN_w8",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer8/EN_w8')))
            bias=tf.get_variable("EN_b8",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer8/EN_b8')))
            EN_conv8=tf.nn.conv2d(dense_cat_67, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv8 = lrelu(EN_conv8)            
            
        dense_cat_678=tf.concat([EN_conv6,EN_conv7,EN_conv8],axis=-1)             
            
####################### Layer 9 #########################   
        with tf.variable_scope('EN_layer9'):
            weights=tf.get_variable("EN_w9",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer9/EN_w9')))
            bias=tf.get_variable("EN_b9",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer9/EN_b9')))
            EN_conv9=tf.nn.conv2d(dense_cat_678, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv9 = lrelu(EN_conv9)             
#################  Spatial Attention 1 ####################   
        SAttent_2_max=tf.reduce_max(EN_conv9, axis=3, keepdims=True)
        SAttent_2_mean=tf.reduce_mean(EN_conv9, axis=3, keepdims=True)
        SAttent_2_cat_mean_max=tf.concat([SAttent_2_max,SAttent_2_mean],axis=-1)        
        with tf.variable_scope('layer2_atten_map'):
            weights=tf.get_variable("w2_atten_map",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_atten_map/w2_atten_map')))
            bias=tf.get_variable("b2_atten_map",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_atten_map/b2_atten_map')))
            conv2_atten_map=tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_atten_map = tf.nn.sigmoid(conv2_atten_map)  
        block_2_atten_out= EN_conv9*conv2_atten_map           
            
        block_23_out = lrelu(block_2_atten_out+block_12_out)             

####################### Layer 10 #########################   
        with tf.variable_scope('EN_layer10'):
            weights=tf.get_variable("EN_w10",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer10/EN_w10')))
            bias=tf.get_variable("EN_b10",initializer=tf.constant(reader_AE.get_tensor('AE_model/EN_layer10/EN_b10')))
            EN_conv10=tf.nn.conv2d(block_23_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv10 = lrelu(EN_conv10) 

    return EN_conv10
    
    
    
def Reconstruct_model(fea,reuse,update_collection=None):
    with tf.variable_scope('Reconstruct_model',reuse=reuse):  
#########################################################     
####################   Decoder   ########################     
######################################################### 

######################################################### 
###################### Layer 1 ##########################         
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('De_layer1_3x3'):
            weights=tf.get_variable("De_w1_3x3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_3x3/De_w1_3x3')))
            bias=tf.get_variable("De_b1_3x3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_3x3/De_b1_3x3')))
            De_conv1_3x3=tf.nn.conv2d(fea, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_3x3 = lrelu(De_conv1_3x3) 
        with tf.variable_scope('De_layer1_5x5'):
            weights=tf.get_variable("De_w1_5x5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_5x5/De_w1_5x5')))
            bias=tf.get_variable("De_b1_5x5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_5x5/De_b1_5x5')))
            De_conv1_5x5=tf.nn.conv2d(fea, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_5x5 = lrelu(De_conv1_5x5) 
        with tf.variable_scope('De_layer1_7x7'):
            weights=tf.get_variable("De_w1_7x7",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_7x7/De_w1_7x7')))
            bias=tf.get_variable("De_b1_7x7",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer1_7x7/De_b1_7x7')))
            De_conv1_7x7=tf.nn.conv2d(fea, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_7x7 = lrelu(De_conv1_7x7)
        De_conv1_cat=tf.concat([De_conv1_3x3,De_conv1_5x5,De_conv1_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_1_max=tf.reduce_max(De_conv1_cat, axis=(1, 2), keepdims=True)
        CAttent_1_mean=tf.reduce_mean(De_conv1_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_CA_max_1'):
            weights=tf.get_variable("w1_CA_max_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_max_1/w1_CA_max_1')))
            bias=tf.get_variable("b1_CA_max_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_max_1/b1_CA_max_1')))
            conv1_CA_max_1=tf.nn.conv2d(CAttent_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_max_1 = tf.nn.relu(conv1_CA_max_1)
        with tf.variable_scope('layer1_CA_mean_1'):
            weights=tf.get_variable("w1_CA_mean_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_mean_1/w1_CA_mean_1')))
            bias=tf.get_variable("b1_CA_mean_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_mean_1/b1_CA_mean_1')))
            conv1_CA_mean_1=tf.nn.conv2d(CAttent_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_mean_1 = tf.nn.relu(conv1_CA_mean_1)            

        with tf.variable_scope('layer1_CA_max_2'):
            weights=tf.get_variable("w1_CA_max_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_max_2/w1_CA_max_2')))
            bias=tf.get_variable("b1_CA_max_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_max_2/b1_CA_max_2')))
            conv1_CA_max_2=tf.nn.conv2d(conv1_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_CA_mean_2'):
            weights=tf.get_variable("w1_CA_mean_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_mean_2/w1_CA_mean_2')))
            bias=tf.get_variable("b1_CA_mean_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer1_CA_mean_2/b1_CA_mean_2')))
            conv1_CA_mean_2=tf.nn.conv2d(conv1_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_CA_atten_map= tf.nn.sigmoid(conv1_CA_max_2+conv1_CA_mean_2)    
        De_conv1_CA_atten_out= De_conv1_cat*conv1_CA_atten_map 

######################################################### 
###################### Layer 2 ##########################         
##################  Multi_Scale 2 #######################       
        with tf.variable_scope('De_layer2_3x3'):
            weights=tf.get_variable("De_w2_3x3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_3x3/De_w2_3x3')))
            bias=tf.get_variable("De_b2_3x3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_3x3/De_b2_3x3')))
            De_conv2_3x3=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_3x3 = lrelu(De_conv2_3x3) 
        with tf.variable_scope('De_layer2_5x5'):
            weights=tf.get_variable("De_w2_5x5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_5x5/De_w2_5x5')))
            bias=tf.get_variable("De_b2_5x5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_5x5/De_b2_5x5')))
            De_conv2_5x5=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_5x5 = lrelu(De_conv2_5x5) 
        with tf.variable_scope('De_layer2_7x7'):
            weights=tf.get_variable("De_w2_7x7",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_7x7/De_w2_7x7')))
            bias=tf.get_variable("De_b2_7x7",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer2_7x7/De_b2_7x7')))
            De_conv2_7x7=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_7x7 = lrelu(De_conv2_7x7)
        De_conv2_cat=tf.concat([De_conv2_3x3,De_conv2_5x5,De_conv2_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_2_max=tf.reduce_max(De_conv2_cat, axis=(1, 2), keepdims=True)
        CAttent_2_mean=tf.reduce_mean(De_conv2_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_CA_max_1'):
            weights=tf.get_variable("w2_CA_max_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_max_1/w2_CA_max_1')))
            bias=tf.get_variable("b2_CA_max_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_max_1/b2_CA_max_1')))
            conv2_CA_max_1=tf.nn.conv2d(CAttent_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_max_1 = tf.nn.relu(conv2_CA_max_1)
        with tf.variable_scope('layer2_CA_mean_1'):
            weights=tf.get_variable("w2_CA_mean_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_mean_1/w2_CA_mean_1')))
            bias=tf.get_variable("b2_CA_mean_1",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_mean_1/b2_CA_mean_1')))
            conv2_CA_mean_1=tf.nn.conv2d(CAttent_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_mean_1 = tf.nn.relu(conv2_CA_mean_1)            

        with tf.variable_scope('layer2_CA_max_2'):
            weights=tf.get_variable("w2_CA_max_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_max_2/w2_CA_max_2')))
            bias=tf.get_variable("b2_CA_max_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_max_2/b2_CA_max_2')))
            conv2_CA_max_2=tf.nn.conv2d(conv2_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_CA_mean_2'):
            weights=tf.get_variable("w2_CA_mean_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_mean_2/w2_CA_mean_2')))
            bias=tf.get_variable("b2_CA_mean_2",initializer=tf.constant(reader_AE.get_tensor('AE_model/layer2_CA_mean_2/b2_CA_mean_2')))
            conv2_CA_mean_2=tf.nn.conv2d(conv2_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_CA_atten_map= tf.nn.sigmoid(conv2_CA_max_2+conv2_CA_mean_2)    
        De_conv2_CA_atten_out= De_conv2_cat*conv2_CA_atten_map

######################################################### 
###################### Layer 3 ##########################
        with tf.variable_scope('De_layer3'):
            weights=tf.get_variable("De_w3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer3/De_w3')))
            bias=tf.get_variable("De_b3",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer3/De_b3')))
            De_conv3=tf.nn.conv2d(De_conv2_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv3 = lrelu(De_conv3) 
            
######################################################### 
###################### Layer 4 ##########################
        with tf.variable_scope('De_layer4'):
            weights=tf.get_variable("De_w4",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer4/De_w4')))
            bias=tf.get_variable("De_b4",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer4/De_b4')))
            De_conv4=tf.nn.conv2d(De_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv4 = lrelu(De_conv4)             

######################################################### 
###################### Layer 5 ##########################
        with tf.variable_scope('De_layer5'):
            weights=tf.get_variable("De_w5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer5/De_w5')))
            bias=tf.get_variable("De_b5",initializer=tf.constant(reader_AE.get_tensor('AE_model/De_layer5/De_b5')))
            De_conv5=tf.nn.conv2d(De_conv4, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv5 = tf.nn.tanh(De_conv5) 
                                            
    return De_conv5
    

def input_setup(index):
    padding=0
    sub_IR_sequence = []
    sub_VI_sequence = []

    
    input_IR=(imread(data_IR[index])-127.5)/127.5
    input_IR=np.lib.pad(input_IR,((padding,padding),(padding,padding)),'edge')
    w,h=input_IR.shape
    input_IR=input_IR.reshape([w,h,1])

    input_VI=(imread(data_VI[index])-127.5)/127.5
    input_VI=np.lib.pad(input_VI,((padding,padding),(padding,padding)),'edge')
    w,h=input_VI.shape
    input_VI=input_VI.reshape([w,h,1])
    
    sub_IR_sequence.append(input_IR)
    sub_VI_sequence.append(input_VI)
    
    train_data_IR= np.asarray(sub_IR_sequence)
    train_data_VI= np.asarray(sub_VI_sequence)

    return train_data_IR,train_data_VI

for idx_num in range(19,20):
  num_epoch=idx_num
  while(num_epoch==idx_num):
  
      reader_AE = tf.train.NewCheckpointReader('./checkpoint/AE_model/AE_model.model-'+ str(100)) 
      reader_Fusion = tf.train.NewCheckpointReader('./checkpoint/Fusion_model/Fusion_model.model-'+ str(num_epoch))     
  
      with tf.name_scope('IR_input'):
          images_IR = tf.placeholder(tf.float32, [1,None,None,None], name='images_IR')
      with tf.name_scope('VI_input'):
          images_VI = tf.placeholder(tf.float32, [1,None,None,None], name='images_VI')
          
      with tf.name_scope('input'):
          input_image_IR  =images_IR
          input_image_VI =images_VI

      with tf.name_scope('fusion'):
          Encoder_fea_IR=Extract_model(input_image_IR,reuse=False)
          Encoder_fea_VI=Extract_model(input_image_VI,reuse=True,update_collection=None)
          Fused_weight_IR,Fused_weight_VI, Fused_weight_bias= fusion_model(Encoder_fea_IR,Encoder_fea_VI,reuse=False)
          Fused_Fea=Fused_weight_IR*Encoder_fea_IR+Fused_weight_VI*Encoder_fea_VI+Fused_weight_bias
          
          fusion_image=Reconstruct_model(Fused_Fea,reuse=False)
  
      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_IR=prepare_data('data/Test_data/Test_IR')
          data_VI=prepare_data('data/Test_data/Test_VI')

          for i in range(len(data_IR)):
              train_data_IR,train_data_VI=input_setup(i)
              start=time.time()
              result =sess.run(fusion_image,feed_dict={images_IR: train_data_IR,images_VI: train_data_VI})
              result=result*127.5+127.5
              result = result.squeeze()
              end=time.time()
              image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
              if not os.path.exists(image_path):
                  os.makedirs(image_path)              
              image_path = os.path.join(image_path,str(i+1)+".bmp")
              imsave(result, image_path)
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1
