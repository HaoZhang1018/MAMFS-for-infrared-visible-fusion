# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup_image, 
  imsave,
  sobel_gradient,
  lrelu,
  l2_norm,
  linear_map,
  lpls_gradient,
  lpls_gradient_2,
  sobel_gradient_4
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class Fusion_model(object):

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
        
        
        self.reader = tf.train.NewCheckpointReader('./checkpoint/AE_model/AE_model.model-'+ str(100))
        
        
    with tf.name_scope('input'):
        self.input_image_IR = self.images_IR                                       
        self.input_image_VI = self.images_VI


    with tf.name_scope('AE'): 
        self.Encoder_fea_IR, self.recon_image_IR=self.AE_model(self.input_image_IR,reuse=False)
        self.Encoder_fea_VI, self.recon_image_VI=self.AE_model(self.input_image_VI,reuse=True,update_collection=None)

    with tf.name_scope('fusion'): 
    
        self.fused_weight_IR,self.fused_weight_VI,self.fused_bias=self.fusion_model(self.Encoder_fea_IR,self.Encoder_fea_VI,reuse=False)
        self.fused_fea=(self.fused_weight_IR*self.Encoder_fea_IR) + (self.fused_weight_VI*self.Encoder_fea_VI) + self.fused_bias
        
        
    with tf.name_scope('d_loss'):
        prob_true_ir=self.discriminator(self.Encoder_fea_IR,reuse=False)
        prob_true_vi=self.discriminator(self.Encoder_fea_VI,reuse=True,update_collection='NO_OPS')
        prob_fake=self.discriminator(self.fused_fea,reuse=True,update_collection='NO_OPS')
        
        True_loss_ir=tf.reduce_mean(tf.square(tf.expand_dims(prob_true_ir[:,0],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.8,maxval=1.0))+tf.square(tf.expand_dims(prob_true_ir[:,1],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.0,maxval=0.2)))
        True_loss_vi=tf.reduce_mean(tf.square(tf.expand_dims(prob_true_vi[:,0],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.0,maxval=0.2))+tf.square(tf.expand_dims(prob_true_vi[:,1],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.8,maxval=1.0)))                
        Fake_loss=tf.reduce_mean(tf.square(tf.expand_dims(prob_fake[:,0],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))+tf.reduce_mean(tf.square(tf.expand_dims(prob_fake[:,1],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))
        
        self.dis_loss=0.25*(True_loss_ir + True_loss_vi) + 0.5*Fake_loss
        
        tf.summary.scalar('loss_d',self.dis_loss)
                
    with tf.name_scope('g_loss'):
        self.g_loss_dis=tf.reduce_mean(tf.square(tf.expand_dims(prob_fake[:,0],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.4,maxval=0.6,dtype=tf.float32)))+tf.reduce_mean(tf.square(tf.expand_dims(prob_fake[:,1],-1)-tf.random_uniform(shape=[self.batch_size,1],minval=0.4,maxval=0.6,dtype=tf.float32))) 
        self.g_loss_total=10*self.g_loss_dis
        
        tf.summary.scalar('g_loss_dis',self.g_loss_dis)                      
        tf.summary.scalar('g_loss_total',self.g_loss_total)   

    self.saver = tf.train.Saver(max_to_keep=200)
    with tf.name_scope('image'):
        tf.summary.image('input_image_IR',tf.expand_dims(self.input_image_IR[1,:,:,:],0))
        tf.summary.image('input_image_VI',tf.expand_dims(self.input_image_VI[1,:,:,:],0))
        tf.summary.image('Encoder_fea_IR',tf.expand_dims(self.Encoder_fea_IR[1,:,:,0:3],0))
        tf.summary.image('Encoder_fea_VI',tf.expand_dims(self.Encoder_fea_VI[1,:,:,0:3],0))
        tf.summary.image('fused_fea',tf.expand_dims(self.fused_fea[1,:,:,0:3],0)) 


  def train(self, config):
    if config.is_train:
      input_setup_image(self.sess, config,"data/Train_data/Train_IR")
      input_setup_image(self.sess, config,"data/Train_data/Train_VI")

    if config.is_train:     
      data_dir_IR = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_IR","train.h5")
      data_dir_VI = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_VI","train.h5")
      
    train_data_IR= read_data(data_dir_IR)
    train_data_VI= read_data(data_dir_VI)
    
    F_vars = tf.trainable_variables()    
    self.dis_vars = [var for var in F_vars if 'discriminator' in var.name]
    print(self.dis_vars)
    self.fusion_vars = [var for var in F_vars if 'fusion_model' in var.name]
    print(self.fusion_vars)
    

    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.fusion_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.dis_loss,var_list=self.dis_vars)
                      
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
          for i in range(2):
            _, err_d= self.sess.run([self.train_discriminator_op, self.dis_loss], feed_dict={self.images_IR: batch_images_IR,self.images_VI: batch_images_VI})
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_IR: batch_images_IR,self.images_VI: batch_images_VI})
          self.train_writer.add_summary(summary_str,counter)        

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f],loss_d:[%.8f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d, err_g))
        self.save(config.checkpoint_dir, ep)

  def fusion_model(self,fea_IR,fea_VI, reuse,update_collection=None):
    with tf.variable_scope('fusion_model',reuse=reuse): 
        input_fea_cat = tf.concat([fea_IR,fea_VI],axis=-1)    
######################################################### 
#################### Layer 1 ############################     
        with tf.variable_scope('Fusion_layer1'):
            weights=tf.get_variable("Fusion_w1",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b1",[64],initializer=tf.constant_initializer(0.0))
            Fusion_conv1=tf.nn.conv2d(input_fea_cat, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv1 = lrelu(Fusion_conv1) 
            
######################################################### 
#################### Layer 2 ############################      
        with tf.variable_scope('Fusion_layer2'):
            weights=tf.get_variable("Fusion_w2",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b2",[128],initializer=tf.constant_initializer(0.0))
            Fusion_conv2=tf.nn.conv2d(Fusion_conv1, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv2 = lrelu(Fusion_conv2)       
######################################################### 
#################### Layer 3 ############################       
        with tf.variable_scope('Fusion_layer3'):
            weights=tf.get_variable("Fusion_w3",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b3",[64],initializer=tf.constant_initializer(0.0))
            Fusion_conv3=tf.nn.conv2d(Fusion_conv2, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv3 = lrelu(Fusion_conv3)  
            
######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_ir'):
            weights=tf.get_variable("Fusion_w4_ir",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b4_ir",[32],initializer=tf.constant_initializer(0.0))
            Fusion_conv4_ir =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_ir = lrelu(Fusion_conv4_ir)

######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_vi'):
            weights=tf.get_variable("Fusion_w4_vi",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b4_vi",[32],initializer=tf.constant_initializer(0.0))
            Fusion_conv4_vi =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_vi = lrelu(Fusion_conv4_vi)
            
######################################################### 
#################### Layer 4  #########################            
        with tf.variable_scope('Fusion_layer4_bias'):
            weights=tf.get_variable("Fusion_w4_bias",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b4_bias",[32],initializer=tf.constant_initializer(0.0))
            Fusion_conv4_bias =tf.nn.conv2d(Fusion_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv4_bias = lrelu(Fusion_conv4_bias)
            
######################################################### 
#################### Layer 5 #########################            
        with tf.variable_scope('Fusion_layer5_ir'):
            weights=tf.get_variable("Fusion_w5_ir",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b5_ir",[16],initializer=tf.constant_initializer(0.0))
            Fusion_conv5_ir =tf.nn.conv2d(Fusion_conv4_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_ir = tf.nn.sigmoid(Fusion_conv5_ir)           

######################################################### 
#################### Layer 6 #########################            
        with tf.variable_scope('Fusion_layer5_vi'):
            weights=tf.get_variable("Fusion_w5_vi",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b5_vi",[16],initializer=tf.constant_initializer(0.0))
            Fusion_conv5_vi =tf.nn.conv2d(Fusion_conv4_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_vi = tf.nn.sigmoid(Fusion_conv5_vi)  

######################################################### 
#################### Layer 6 #########################            
        with tf.variable_scope('Fusion_layer5_bias'):
            weights=tf.get_variable("Fusion_w5_bias",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Fusion_b5_bias",[16],initializer=tf.constant_initializer(0.0))
            Fusion_conv5_bias =tf.nn.conv2d(Fusion_conv4_bias, weights, strides=[1,1,1,1], padding='SAME') + bias
            Fusion_conv5_bias = lrelu(Fusion_conv5_bias)  

                                                                                                        
    return Fusion_conv5_ir, Fusion_conv5_vi,Fusion_conv5_bias
    

  def discriminator(self,fea,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        with tf.variable_scope('Dis_layer1'):
            weights=tf.get_variable("Dis_w1",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Dis_b1",[32],initializer=tf.constant_initializer(0.0))
            Dis_conv1=tf.nn.conv2d(fea, weights, strides=[1,2,2,1], padding='VALID') + bias
            Dis_conv1 = lrelu(Dis_conv1)

        with tf.variable_scope('Dis_layer2'):
            weights=tf.get_variable("Dis_w2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Dis_b2",[64],initializer=tf.constant_initializer(0.0))
            Dis_conv2= tf.nn.conv2d(Dis_conv1, weights, strides=[1,2,2,1], padding='VALID') + bias
            Dis_conv2 = lrelu(Dis_conv2)

        with tf.variable_scope('Dis_layer3'):
            weights=tf.get_variable("Dis_w3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Dis_b3",[128],initializer=tf.constant_initializer(0.0))
            Dis_conv3= tf.nn.conv2d(Dis_conv2, weights, strides=[1,2,2,1], padding='VALID') + bias
            Dis_conv3=lrelu(Dis_conv3)

        with tf.variable_scope('Dis_layer4'):
            weights=tf.get_variable("Dis_w4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Dis_b4",[256],initializer=tf.constant_initializer(0.0))
            Dis_conv4= tf.nn.conv2d(Dis_conv3, weights, strides=[1,2,2,1], padding='VALID') + bias
            Dis_conv4=lrelu(Dis_conv4)
                        
            Dis_conv_vector = tf.reshape(Dis_conv4,[self.batch_size,4*4*256])
            
        with tf.variable_scope('Dis_layer5'):
            weights=tf.get_variable("Dis_w5",[4*4*256,2],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("Dis_b5",[2],initializer=tf.constant_initializer(0.0))
            Dis_prob=tf.matmul(Dis_conv_vector, weights) + bias                      
    return Dis_prob


  def AE_model(self,img,reuse,update_collection=None):
    with tf.variable_scope('AE_model',reuse=reuse):   
#########################################################     
####################   Encoder   ########################     
#########################################################     
    
######################################################### 
###################### Layer 1 ##########################      
        with tf.variable_scope('EN_layer1'):
            weights=tf.get_variable("EN_w1",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer1/EN_w1')))
            bias=tf.get_variable("EN_b1",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer1/EN_b1')))
            EN_conv1 = tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv1 = lrelu(EN_conv1)

####################### Block 1 ######################### 
####################### Layer 2 #########################   
        with tf.variable_scope('EN_layer2'):
            weights=tf.get_variable("EN_w2",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer2/EN_w2')))
            bias=tf.get_variable("EN_b2",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer2/EN_b2')))
            EN_conv2=tf.nn.conv2d(EN_conv1, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv2 = lrelu(EN_conv2)   
####################### Layer 3 #########################   
        with tf.variable_scope('EN_layer3'):
            weights=tf.get_variable("EN_w3",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer3/EN_w3')))
            bias=tf.get_variable("EN_b3",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer3/EN_b3')))
            EN_conv3=tf.nn.conv2d(EN_conv2, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv3 = lrelu(EN_conv3) 
                        
        dense_cat_23=tf.concat([EN_conv2,EN_conv3],axis=-1)             
            
####################### Layer 4 #########################   
        with tf.variable_scope('EN_layer4'):
            weights=tf.get_variable("EN_w4",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer4/EN_w4')))
            bias=tf.get_variable("EN_b4",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer4/EN_b4')))
            EN_conv4=tf.nn.conv2d(dense_cat_23, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv4 = lrelu(EN_conv4)            
            
        dense_cat_234=tf.concat([EN_conv2,EN_conv3,EN_conv4],axis=-1)             
            
####################### Layer 5 #########################   
        with tf.variable_scope('EN_layer5'):
            weights=tf.get_variable("EN_w5",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer5/EN_w5')))
            bias=tf.get_variable("EN_b5",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer5/EN_b5')))
            EN_conv5=tf.nn.conv2d(dense_cat_234, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv5 = lrelu(EN_conv5)             
#################  Spatial Attention 1 ####################   
        SAttent_1_max=tf.reduce_max(EN_conv5, axis=3, keepdims=True)
        SAttent_1_mean=tf.reduce_mean(EN_conv5, axis=3, keepdims=True)
        SAttent_1_cat_mean_max=tf.concat([SAttent_1_max,SAttent_1_mean],axis=-1)        
        with tf.variable_scope('layer1_atten_map'):
            weights=tf.get_variable("w1_atten_map",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_atten_map/w1_atten_map')))
            bias=tf.get_variable("b1_atten_map",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_atten_map/b1_atten_map')))
            conv1_atten_map=tf.nn.conv2d(SAttent_1_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_atten_map = tf.nn.sigmoid(conv1_atten_map)  
        block_1_atten_out= EN_conv5*conv1_atten_map           
            
        block_12_out = lrelu(block_1_atten_out+EN_conv1)           
            
            
####################### Block 2 ######################### 
####################### Layer 6 #########################   
        with tf.variable_scope('EN_layer6'):
            weights=tf.get_variable("EN_w6",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer6/EN_w6')))
            bias=tf.get_variable("EN_b6",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer6/EN_b6')))
            EN_conv6=tf.nn.conv2d(block_12_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv6 = lrelu(EN_conv6)   
####################### Layer 7 #########################   
        with tf.variable_scope('EN_layer7'):
            weights=tf.get_variable("EN_w7",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer7/EN_w7')))
            bias=tf.get_variable("EN_b7",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer7/EN_b7')))
            EN_conv7=tf.nn.conv2d(EN_conv6, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv7 = lrelu(EN_conv7) 
                        
        dense_cat_67=tf.concat([EN_conv6,EN_conv7],axis=-1)             
            
####################### Layer 8 #########################   
        with tf.variable_scope('EN_layer8'):
            weights=tf.get_variable("EN_w8",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer8/EN_w8')))
            bias=tf.get_variable("EN_b8",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer8/EN_b8')))
            EN_conv8=tf.nn.conv2d(dense_cat_67, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv8 = lrelu(EN_conv8)            
            
        dense_cat_678=tf.concat([EN_conv6,EN_conv7,EN_conv8],axis=-1)             
            
####################### Layer 9 #########################   
        with tf.variable_scope('EN_layer9'):
            weights=tf.get_variable("EN_w9",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer9/EN_w9')))
            bias=tf.get_variable("EN_b9",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer9/EN_b9')))
            EN_conv9=tf.nn.conv2d(dense_cat_678, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv9 = lrelu(EN_conv9)             
#################  Spatial Attention 1 ####################   
        SAttent_2_max=tf.reduce_max(EN_conv9, axis=3, keepdims=True)
        SAttent_2_mean=tf.reduce_mean(EN_conv9, axis=3, keepdims=True)
        SAttent_2_cat_mean_max=tf.concat([SAttent_2_max,SAttent_2_mean],axis=-1)        
        with tf.variable_scope('layer2_atten_map'):
            weights=tf.get_variable("w2_atten_map",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_atten_map/w2_atten_map')))
            bias=tf.get_variable("b2_atten_map",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_atten_map/b2_atten_map')))
            conv2_atten_map=tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_atten_map = tf.nn.sigmoid(conv2_atten_map)  
        block_2_atten_out= EN_conv9*conv2_atten_map           
            
        block_23_out = lrelu(block_2_atten_out+block_12_out)             

####################### Layer 10 #########################   
        with tf.variable_scope('EN_layer10'):
            weights=tf.get_variable("EN_w10",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer10/EN_w10')))
            bias=tf.get_variable("EN_b10",initializer=tf.constant(self.reader.get_tensor('AE_model/EN_layer10/EN_b10')))
            EN_conv10=tf.nn.conv2d(block_23_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            EN_conv10 = lrelu(EN_conv10) 




#########################################################     
####################   Decoder   ########################     
######################################################### 

######################################################### 
###################### Layer 1 ##########################         
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('De_layer1_3x3'):
            weights=tf.get_variable("De_w1_3x3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_3x3/De_w1_3x3')))
            bias=tf.get_variable("De_b1_3x3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_3x3/De_b1_3x3')))
            De_conv1_3x3=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_3x3 = lrelu(De_conv1_3x3) 
        with tf.variable_scope('De_layer1_5x5'):
            weights=tf.get_variable("De_w1_5x5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_5x5/De_w1_5x5')))
            bias=tf.get_variable("De_b1_5x5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_5x5/De_b1_5x5')))
            De_conv1_5x5=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_5x5 = lrelu(De_conv1_5x5) 
        with tf.variable_scope('De_layer1_7x7'):
            weights=tf.get_variable("De_w1_7x7",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_7x7/De_w1_7x7')))
            bias=tf.get_variable("De_b1_7x7",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer1_7x7/De_b1_7x7')))
            De_conv1_7x7=tf.nn.conv2d(EN_conv10, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv1_7x7 = lrelu(De_conv1_7x7)
        De_conv1_cat=tf.concat([De_conv1_3x3,De_conv1_5x5,De_conv1_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_1_max=tf.reduce_max(De_conv1_cat, axis=(1, 2), keepdims=True)
        CAttent_1_mean=tf.reduce_mean(De_conv1_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_CA_max_1'):
            weights=tf.get_variable("w1_CA_max_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_max_1/w1_CA_max_1')))
            bias=tf.get_variable("b1_CA_max_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_max_1/b1_CA_max_1')))
            conv1_CA_max_1=tf.nn.conv2d(CAttent_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_max_1 = tf.nn.relu(conv1_CA_max_1)
        with tf.variable_scope('layer1_CA_mean_1'):
            weights=tf.get_variable("w1_CA_mean_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_mean_1/w1_CA_mean_1')))
            bias=tf.get_variable("b1_CA_mean_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_mean_1/b1_CA_mean_1')))
            conv1_CA_mean_1=tf.nn.conv2d(CAttent_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_CA_mean_1 = tf.nn.relu(conv1_CA_mean_1)            

        with tf.variable_scope('layer1_CA_max_2'):
            weights=tf.get_variable("w1_CA_max_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_max_2/w1_CA_max_2')))
            bias=tf.get_variable("b1_CA_max_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_max_2/b1_CA_max_2')))
            conv1_CA_max_2=tf.nn.conv2d(conv1_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_CA_mean_2'):
            weights=tf.get_variable("w1_CA_mean_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_mean_2/w1_CA_mean_2')))
            bias=tf.get_variable("b1_CA_mean_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer1_CA_mean_2/b1_CA_mean_2')))
            conv1_CA_mean_2=tf.nn.conv2d(conv1_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_CA_atten_map= tf.nn.sigmoid(conv1_CA_max_2+conv1_CA_mean_2)    
        De_conv1_CA_atten_out= De_conv1_cat*conv1_CA_atten_map 

######################################################### 
###################### Layer 2 ##########################         
##################  Multi_Scale 2 #######################       
        with tf.variable_scope('De_layer2_3x3'):
            weights=tf.get_variable("De_w2_3x3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_3x3/De_w2_3x3')))
            bias=tf.get_variable("De_b2_3x3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_3x3/De_b2_3x3')))
            De_conv2_3x3=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_3x3 = lrelu(De_conv2_3x3) 
        with tf.variable_scope('De_layer2_5x5'):
            weights=tf.get_variable("De_w2_5x5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_5x5/De_w2_5x5')))
            bias=tf.get_variable("De_b2_5x5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_5x5/De_b2_5x5')))
            De_conv2_5x5=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_5x5 = lrelu(De_conv2_5x5) 
        with tf.variable_scope('De_layer2_7x7'):
            weights=tf.get_variable("De_w2_7x7",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_7x7/De_w2_7x7')))
            bias=tf.get_variable("De_b2_7x7",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer2_7x7/De_b2_7x7')))
            De_conv2_7x7=tf.nn.conv2d(De_conv1_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv2_7x7 = lrelu(De_conv2_7x7)
        De_conv2_cat=tf.concat([De_conv2_3x3,De_conv2_5x5,De_conv2_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_2_max=tf.reduce_max(De_conv2_cat, axis=(1, 2), keepdims=True)
        CAttent_2_mean=tf.reduce_mean(De_conv2_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_CA_max_1'):
            weights=tf.get_variable("w2_CA_max_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_max_1/w2_CA_max_1')))
            bias=tf.get_variable("b2_CA_max_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_max_1/b2_CA_max_1')))
            conv2_CA_max_1=tf.nn.conv2d(CAttent_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_max_1 = tf.nn.relu(conv2_CA_max_1)
        with tf.variable_scope('layer2_CA_mean_1'):
            weights=tf.get_variable("w2_CA_mean_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_mean_1/w2_CA_mean_1')))
            bias=tf.get_variable("b2_CA_mean_1",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_mean_1/b2_CA_mean_1')))
            conv2_CA_mean_1=tf.nn.conv2d(CAttent_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_CA_mean_1 = tf.nn.relu(conv2_CA_mean_1)            

        with tf.variable_scope('layer2_CA_max_2'):
            weights=tf.get_variable("w2_CA_max_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_max_2/w2_CA_max_2')))
            bias=tf.get_variable("b2_CA_max_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_max_2/b2_CA_max_2')))
            conv2_CA_max_2=tf.nn.conv2d(conv2_CA_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_CA_mean_2'):
            weights=tf.get_variable("w2_CA_mean_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_mean_2/w2_CA_mean_2')))
            bias=tf.get_variable("b2_CA_mean_2",initializer=tf.constant(self.reader.get_tensor('AE_model/layer2_CA_mean_2/b2_CA_mean_2')))
            conv2_CA_mean_2=tf.nn.conv2d(conv2_CA_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_CA_atten_map= tf.nn.sigmoid(conv2_CA_max_2+conv2_CA_mean_2)    
        De_conv2_CA_atten_out= De_conv2_cat*conv2_CA_atten_map

######################################################### 
###################### Layer 3 ##########################
        with tf.variable_scope('De_layer3'):
            weights=tf.get_variable("De_w3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer3/De_w3')))
            bias=tf.get_variable("De_b3",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer3/De_b3')))
            De_conv3=tf.nn.conv2d(De_conv2_CA_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv3 = lrelu(De_conv3) 
            
######################################################### 
###################### Layer 4 ##########################
        with tf.variable_scope('De_layer4'):
            weights=tf.get_variable("De_w4",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer4/De_w4')))
            bias=tf.get_variable("De_b4",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer4/De_b4')))
            De_conv4=tf.nn.conv2d(De_conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv4 = lrelu(De_conv4)             

######################################################### 
###################### Layer 5 ##########################
        with tf.variable_scope('De_layer5'):
            weights=tf.get_variable("De_w5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer5/De_w5')))
            bias=tf.get_variable("De_b5",initializer=tf.constant(self.reader.get_tensor('AE_model/De_layer5/De_b5')))
            De_conv5=tf.nn.conv2d(De_conv4, weights, strides=[1,1,1,1], padding='SAME') + bias
            De_conv5 = tf.nn.tanh(De_conv5) 
                                            
    return EN_conv10, De_conv5




  def save(self, checkpoint_dir, step):
    model_name = "Fusion_model.model"
    model_dir = "%s" % ("Fusion_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s" % ("Fusion_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
