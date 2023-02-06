# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup_image,  
  imsave,
  merge,
  sobel_gradient,
  lrelu,
  l2_norm,
  linear_map,
  lpls_gradient,
  lpls_gradient_4,
  sobel_gradient_4
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class AE_model(object):

  def __init__(self, 
               sess, 
               image_size=80,
               batch_size=48,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size =  image_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
  ########## IR  Input ###################### 
    with tf.name_scope('IR_input'):
        self.images_IR = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_IR')
  ########## VI  Input ######################                         
    with tf.name_scope('VI_input'):
        self.images_VI = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_VI')

    with tf.name_scope('input'): 
        self.input_image_IR  = self.images_IR                                       
        self.input_image_VI = self.images_VI


    with tf.name_scope('AE'): 
        self.encoder_fea_ir,self.decoder_image_ir=self.AE_model(self.input_image_IR,reuse=False)
        self.encoder_fea_vi,self.decoder_image_vi=self.AE_model(self.input_image_VI,reuse=True,update_collection='NO_OPS')

    with tf.name_scope('Gradient'): 
        self.IR_gradient_x,self.IR_gradient_y=sobel_gradient(self.input_image_IR) 
        self.DE_IR_gradient_x,self.DE_IR_gradient_y=sobel_gradient(self.decoder_image_ir)  
             
        self.VI_gradient_x,self.VI_gradient_y=sobel_gradient(self.input_image_VI) 
        self.DE_VI_gradient_x,self.DE_VI_gradient_y=sobel_gradient(self.decoder_image_vi)



    with tf.name_scope('AE_loss'):
        self.AE_loss_ir_int  = tf.reduce_mean(tf.abs(self.input_image_IR - self.decoder_image_ir))
        self.AE_loss_vi_int  = tf.reduce_mean(tf.abs(self.input_image_VI - self.decoder_image_vi))
        self.AE_loss_int  = self.AE_loss_ir_int + self.AE_loss_vi_int
                
        self.AE_loss_ir_grad_x  = tf.reduce_mean(tf.abs(self.IR_gradient_x - self.DE_IR_gradient_x))
        self.AE_loss_ir_grad_y  = tf.reduce_mean(tf.abs(self.IR_gradient_y - self.DE_IR_gradient_y))
        self.AE_loss_ir_grad = self.AE_loss_ir_grad_x + self.AE_loss_ir_grad_y        
        self.AE_loss_vi_grad_x  = tf.reduce_mean(tf.abs(self.VI_gradient_x - self.DE_VI_gradient_x))
        self.AE_loss_vi_grad_y  = tf.reduce_mean(tf.abs(self.VI_gradient_y - self.DE_VI_gradient_y))
        self.AE_loss_vi_grad = self.AE_loss_vi_grad_x + self.AE_loss_vi_grad_y                
        self.AE_loss_grad =  self.AE_loss_ir_grad + self.AE_loss_vi_grad
        
        self.AE_loss_total=100*(self.AE_loss_int + 10* self.AE_loss_grad)
        
            
        tf.summary.scalar('AE_loss_ir_int',self.AE_loss_ir_int)
        tf.summary.scalar('AE_loss_vi_int',self.AE_loss_vi_int)
        tf.summary.scalar('AE_loss_int',self.AE_loss_int) 
                        
        tf.summary.scalar('AE_loss_ir_grad',self.AE_loss_ir_grad)
        tf.summary.scalar('AE_loss_vi_grad',self.AE_loss_vi_grad)
        tf.summary.scalar('AE_loss_grad',self.AE_loss_grad)
        
        tf.summary.scalar('AE_loss_total',self.AE_loss_total)        
    self.saver = tf.train.Saver(max_to_keep=100)
    
    
    with tf.name_scope('image'):
        tf.summary.image('input_image_IR',tf.expand_dims(self.input_image_IR[1,:,:,:],0))  
        tf.summary.image('decoder_image_ir',tf.expand_dims(self.decoder_image_ir[1,:,:,:],0))  
        tf.summary.image('input_image_VI',tf.expand_dims(self.input_image_VI[1,:,:,:],0))
        tf.summary.image('decoder_image_vi',tf.expand_dims(self.decoder_image_vi[1,:,:,:],0))

    
  def train(self, config):
    if config.is_train:
      input_setup_image(self.sess,config,"data/Train_data/Train_IR")
      input_setup_image(self.sess,config,"data/Train_data/Train_VI")    
      data_dir_IR = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_IR","train.h5")
      data_dir_VI = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_VI","train.h5")
      
    train_data_IR= read_data(data_dir_IR)
    train_data_VI= read_data(data_dir_VI)

    
    AE_vars = tf.trainable_variables()
    self.AEencoder_vars = [var for var in AE_vars if 'AE_model' in var.name]
    print(self.AEencoder_vars)
    with tf.name_scope('train_step'):
        self.train_AE_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.AE_loss_total,var_list=self.AEencoder_vars)        
    self.summary_op = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)    
    tf.initialize_all_variables().run()    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_IR) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_IR = train_data_IR[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_VI = train_data_VI[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1          
          _, err_AE,summary_str= self.sess.run([self.train_AE_op, self.AE_loss_total,self.summary_op], feed_dict={self.images_IR: batch_images_IR,self.images_VI: batch_images_VI})
          self.train_writer.add_summary(summary_str,counter)
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f],loss_AE:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_AE))
        self.save(config.checkpoint_dir, ep)


  def AE_model(self,img,reuse,update_collection=None):
    with tf.variable_scope('AE_model',reuse=reuse):   
#########################################################     
####################   Encoder   ########################     
#########################################################     
    
######################################################### 
###################### Layer 1 ##########################      
        with tf.variable_scope('EN_layer1'):
            weights=tf.get_variable("EN_w1",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b1",[16],initializer=tf.constant_initializer(0.0))
            EN_conv1 = tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv1 = lrelu(EN_conv1)

####################### Block 1 ######################### 
####################### Layer 2 #########################   
        with tf.variable_scope('EN_layer2'):
            weights=tf.get_variable("EN_w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b2",[16],initializer=tf.constant_initializer(0.0))
            EN_conv2=tf.nn.conv2d(EN_conv1, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv2 = lrelu(EN_conv2)   
####################### Layer 3 #########################   
        with tf.variable_scope('EN_layer3'):
            weights=tf.get_variable("EN_w3",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b3",[16],initializer=tf.constant_initializer(0.0))
            EN_conv3=tf.nn.conv2d(EN_conv2, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv3 = lrelu(EN_conv3) 
                        
        dense_cat_23=tf.concat([EN_conv2,EN_conv3],axis=-1)             
            
####################### Layer 4 #########################   
        with tf.variable_scope('EN_layer4'):
            weights=tf.get_variable("EN_w4",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b4",[16],initializer=tf.constant_initializer(0.0))
            EN_conv4=tf.nn.conv2d(dense_cat_23, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv4 = lrelu(EN_conv4)            
            
        dense_cat_234=tf.concat([EN_conv2,EN_conv3,EN_conv4],axis=-1)             
            
####################### Layer 5 #########################   
        with tf.variable_scope('EN_layer5'):
            weights=tf.get_variable("EN_w5",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b5",[16],initializer=tf.constant_initializer(0.0))
            EN_conv5=tf.nn.conv2d(dense_cat_234, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv5 = lrelu(EN_conv5)             
#################  Spatial Attention 1 ####################   
        SAttent_1_max=tf.reduce_max(EN_conv5, axis=3, keepdims=True)
        SAttent_1_mean=tf.reduce_mean(EN_conv5, axis=3, keepdims=True)
        SAttent_1_cat_mean_max=tf.concat([SAttent_1_max,SAttent_1_mean],axis=-1)        
        with tf.variable_scope('layer1_atten_map'):
            weights=tf.get_variable("w1_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv1_atten_map=tf.nn.conv2d(SAttent_1_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_atten_map = tf.nn.sigmoid(conv1_atten_map)  
        block_1_atten_out= EN_conv5*conv1_atten_map           
            
        block_12_out = lrelu(block_1_atten_out+EN_conv1)           
            
            
####################### Block 2 ######################### 
####################### Layer 6 #########################   
        with tf.variable_scope('EN_layer6'):
            weights=tf.get_variable("EN_w6",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b6",[16],initializer=tf.constant_initializer(0.0))
            EN_conv6=tf.nn.conv2d(block_12_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv6 = lrelu(EN_conv6)   
####################### Layer 7 #########################   
        with tf.variable_scope('EN_layer7'):
            weights=tf.get_variable("EN_w7",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b7",[16],initializer=tf.constant_initializer(0.0))
            EN_conv7=tf.nn.conv2d(EN_conv6, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv7 = lrelu(EN_conv7) 
                        
        dense_cat_67=tf.concat([EN_conv6,EN_conv7],axis=-1)             
            
####################### Layer 8 #########################   
        with tf.variable_scope('EN_layer8'):
            weights=tf.get_variable("EN_w8",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b8",[16],initializer=tf.constant_initializer(0.0))
            EN_conv8=tf.nn.conv2d(dense_cat_67, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv8 = lrelu(EN_conv8)            
            
        dense_cat_678=tf.concat([EN_conv6,EN_conv7,EN_conv8],axis=-1)             
            
####################### Layer 9 #########################   
        with tf.variable_scope('EN_layer9'):
            weights=tf.get_variable("EN_w9",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b9",[16],initializer=tf.constant_initializer(0.0))
            EN_conv9=tf.nn.conv2d(dense_cat_678, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv9 = lrelu(EN_conv9)             
#################  Spatial Attention 1 ####################   
        SAttent_2_max=tf.reduce_max(EN_conv9, axis=3, keepdims=True)
        SAttent_2_mean=tf.reduce_mean(EN_conv9, axis=3, keepdims=True)
        SAttent_2_cat_mean_max=tf.concat([SAttent_2_max,SAttent_2_mean],axis=-1)        
        with tf.variable_scope('layer2_atten_map'):
            weights=tf.get_variable("w2_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv2_atten_map=tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_atten_map = tf.nn.sigmoid(conv2_atten_map)  
        block_2_atten_out= EN_conv9*conv2_atten_map           
            
        block_23_out = lrelu(block_2_atten_out+block_12_out)             

####################### Layer 10 #########################   
        with tf.variable_scope('EN_layer10'):
            weights=tf.get_variable("EN_w10",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("EN_b10",[16],initializer=tf.constant_initializer(0.0))
            EN_conv10=tf.nn.conv2d(block_23_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv10 = lrelu(EN_conv10) 




#########################################################     
####################   Decoder   ########################     
######################################################### 

######################################################### 
###################### Layer 1 ##########################         
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('De_layer1_3x3'):
            weights=tf.get_variable("De_w1_3x3",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b1_3x3",[16],initializer=tf.constant_initializer(0.0))
            De_conv1_3x3=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_3x3 = lrelu(De_conv1_3x3) 
        with tf.variable_scope('De_layer1_5x5'):
            weights=tf.get_variable("De_w1_5x5",[5,5,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b1_5x5",[16],initializer=tf.constant_initializer(0.0))
            De_conv1_5x5=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_5x5 = lrelu(De_conv1_5x5) 
        with tf.variable_scope('De_layer1_7x7'):
            weights=tf.get_variable("De_w1_7x7",[7,7,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b1_7x7",[16],initializer=tf.constant_initializer(0.0))
            De_conv1_7x7=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_7x7 = lrelu(De_conv1_7x7)
        De_conv1_cat=tf.concat([De_conv1_3x3,De_conv1_5x5,De_conv1_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_1_max=tf.reduce_max(De_conv1_cat, axis=(1, 2), keepdims=True)
        CAttent_1_mean=tf.reduce_mean(De_conv1_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_CA_max_1'):
            weights=tf.get_variable("w1_CA_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_CA_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_CA_max_1=tf.nn.conv2d(CAttent_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_max_1 = tf.nn.relu(conv1_CA_max_1)
        with tf.variable_scope('layer1_CA_mean_1'):
            weights=tf.get_variable("w1_CA_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_CA_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_CA_mean_1=tf.nn.conv2d(CAttent_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_mean_1 = tf.nn.relu(conv1_CA_mean_1)            

        with tf.variable_scope('layer1_CA_max_2'):
            weights=tf.get_variable("w1_CA_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_CA_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_CA_max_2=tf.nn.conv2d(conv1_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_CA_mean_2'):
            weights=tf.get_variable("w1_CA_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_CA_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_CA_mean_2=tf.nn.conv2d(conv1_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_CA_atten_map= tf.nn.sigmoid(conv1_CA_max_2+conv1_CA_mean_2)    
        De_conv1_CA_atten_out= De_conv1_cat*conv1_CA_atten_map 

######################################################### 
###################### Layer 2 ##########################         
##################  Multi_Scale 2 #######################       
        with tf.variable_scope('De_layer2_3x3'):
            weights=tf.get_variable("De_w2_3x3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b2_3x3",[16],initializer=tf.constant_initializer(0.0))
            De_conv2_3x3=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_3x3 = lrelu(De_conv2_3x3) 
        with tf.variable_scope('De_layer2_5x5'):
            weights=tf.get_variable("De_w2_5x5",[5,5,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b2_5x5",[16],initializer=tf.constant_initializer(0.0))
            De_conv2_5x5=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_5x5 = lrelu(De_conv2_5x5) 
        with tf.variable_scope('De_layer2_7x7'):
            weights=tf.get_variable("De_w2_7x7",[7,7,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b2_7x7",[16],initializer=tf.constant_initializer(0.0))
            De_conv2_7x7=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_7x7 = lrelu(De_conv2_7x7)
        De_conv2_cat=tf.concat([De_conv2_3x3,De_conv2_5x5,De_conv2_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_2_max=tf.reduce_max(De_conv2_cat, axis=(1, 2), keepdims=True)
        CAttent_2_mean=tf.reduce_mean(De_conv2_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_CA_max_1'):
            weights=tf.get_variable("w2_CA_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_CA_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_CA_max_1=tf.nn.conv2d(CAttent_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_max_1 = tf.nn.relu(conv2_CA_max_1)
        with tf.variable_scope('layer2_CA_mean_1'):
            weights=tf.get_variable("w2_CA_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_CA_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_CA_mean_1=tf.nn.conv2d(CAttent_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_mean_1 = tf.nn.relu(conv2_CA_mean_1)            

        with tf.variable_scope('layer2_CA_max_2'):
            weights=tf.get_variable("w2_CA_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_CA_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_CA_max_2=tf.nn.conv2d(conv2_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_CA_mean_2'):
            weights=tf.get_variable("w2_CA_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_CA_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_CA_mean_2=tf.nn.conv2d(conv2_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_CA_atten_map= tf.nn.sigmoid(conv2_CA_max_2+conv2_CA_mean_2)    
        De_conv2_CA_atten_out= De_conv2_cat*conv2_CA_atten_map

######################################################### 
###################### Layer 3 ##########################
        with tf.variable_scope('De_layer3'):
            weights=tf.get_variable("De_w3",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b3",[12],initializer=tf.constant_initializer(0.0))
            De_conv3=tf.nn.conv2d(De_conv2_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv3 = lrelu(De_conv3) 
            
######################################################### 
###################### Layer 4 ##########################
        with tf.variable_scope('De_layer4'):
            weights=tf.get_variable("De_w4",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b4",[4],initializer=tf.constant_initializer(0.0))
            De_conv4=tf.nn.conv2d(De_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv4 = lrelu(De_conv4)             

######################################################### 
###################### Layer 5 ##########################
        with tf.variable_scope('De_layer5'):
            weights=tf.get_variable("De_w5",[3,3,4,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("De_b5",[1],initializer=tf.constant_initializer(0.0))
            De_conv5=tf.nn.conv2d(De_conv4, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv5 = tf.nn.tanh(De_conv5) 
        
                                    
    return EN_conv10, De_conv5


  def save(self, checkpoint_dir, step):
    model_name = "AE_model.model"
    model_dir = "%s" % ("AE_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s" % ("AE_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
