# -*- coding: utf-8 -*-
"""
To do tasks
-> create one for coloured images
-> optimize the code
"""

"""
-> take in teh images.
-> convert them into translatedif needed
-> convert them to batches
-> make the glimpse sensor
-> make the glimpse network
-> make the core network
-> make the location network
-> make the baseline netork
-> make the action network
-> connect these with each other
-> make a function to evaluate accuracy and draw if needed
-> make a function to train the model
-> make a funciton to test the model
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import random

# All variables defined here
img_size = 60
MNIST_SIZE = 28
batch_size = 64
channels = 1
depth = 3 # number of zooms
min_radius = 8
sensorBandwidth = 12

initLr = 1e-3
lrDecayFreq = 200
lrDecayRate = .999
hl_size = 128
hg_size = 128
g_size  = 256
cell_size = 256             #
cell_out_size = cell_size

time_steps = 5

total_sensor_bandwidth = (sensorBandwidth**2)*depth
channels = 1
loc_sd = 0.22

n_classes = 10              

# training parameters
max_iters = 1000000
SMALL_NUM = 1e-10

# resource prellocation
mean_locs = []              # expectation of locations
sampled_locs = []           # sampled locations ~N(mean_locs[.], loc_sd)
baselines = []              # baseline, the value prediction
glimpse_images = []         # to show in window


dataset = tf_mnist_loader.read_data_sets("mnist_data")

def convertTranslated(images, initImgSize, finalImgSize):
    size_diff = finalImgSize - initImgSize
    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size,2])
    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        # generate and save random coordinates
        randX = random.randint(0, size_diff)
        randY = random.randint(0, size_diff)
        imgCoord[k,:] = np.array([randX, randY])
        # padding
        image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
        newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

    return newimages, imgCoord

# function to plot a random sample from a batch
def plot_sample(batch):
    rand = random.randint(0,batch.shape[0]-1)
    img = batch[rand].reshape((batch.shape[1],batch.shape[2]))
    plt.imshow(img,cmap='gray')
    plt.show()
  
# Function to plot a single image
def plot(img):
    img = img.reshape((img.shape[0],img.shape[1]))
    plt.imshow(img,cmap='gray')
    plt.show()
    
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# The funciton for extracting glimpses
# This one has glimpse_sensor scope
def glimpse_sensor(img,loc):
    with tf.variable_scope("GlimpseSensor"):
        loc = tf.round((loc+1)/2.0*img_size)
        loc = tf.cast(loc,tf.int32)
        img = tf.reshape(img, (batch_size, img_size, img_size, channels))
        max_radius = min_radius*(2**(depth-1))
        offset = 2*max_radius
        img = tf.image.pad_to_bounding_box(img, offset, offset, \
                                                   max_radius * 4 + img_size, max_radius * 4 + img_size)
        zooms = []
        for i in range(batch_size):
            imgZooms = []
            one_img = img[i,:,:,:]
            for j in range(depth):
                 r = int(min_radius * (2 ** (j)))
                 d_raw = 2 * r
                 d = tf.constant(d_raw, shape=[1])
                 d = tf.tile(d,[2])
                 loc_i = loc[i,:]
                 # loc for the image with paddings
                 adjusted_loc = offset + loc_i - r
                 # Take the image without channels
                 one_img = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))
                 # crop image to (d x d)
                 zoom = tf.slice(one_img, adjusted_loc, d)
                 # resize cropped image to (sensorBandwidth x sensorBandwidth)
                 zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
                 zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
                 imgZooms.append(zoom)
            zooms.append(tf.stack(imgZooms))
        zooms = tf.stack(zooms)
        glimpse_images.append(zooms)
        return zooms


# This one has glimpse network scope
def glimpse_network(glimpse_input,loc):
    with tf.variable_scope(g_n_s,auxiliary_name_scope=False) as s1:
        with tf.name_scope(s1.original_name_scope):
            glimpse_input = tf.reshape(glimpse_input,(batch_size,total_sensor_bandwidth))
            glimpse_img_hidden = tf.nn.relu(tf.matmul(glimpse_input, Wg_g_h) + Bg_g_h)
            glimpse_loc_hidden = tf.nn.relu(tf.matmul(loc, Wg_l_h) + Bg_l_h)
            glimpse_combined = tf.nn.relu(tf.matmul(glimpse_img_hidden, Wg_hg_gf1) + tf.matmul(glimpse_loc_hidden, Wg_hl_gf1) + Bg_hlhg_gf1)
    
        return glimpse_combined 



def baseline_network(inpt):
    with tf.variable_scope(b_n_s,auxiliary_name_scope=False) as s1:
        with tf.name_scope(s1.original_name_scope):    
            baseline = tf.sigmoid(tf.matmul(inpt, Wb_h_b) + Bb_h_b)
            baselines.append(baseline)

def loc_network(output):
    with tf.variable_scope(l_n_s,auxiliary_name_scope=False) as s1:
        with tf.name_scope(s1.original_name_scope):    
            # compute the next location, then impose noise
            mean_loc = tf.matmul(output, Wl_h_l) + Bl_h_l
            mean_loc = tf.clip_by_value(mean_loc, -1, 1)
            # mean_loc = tf.stop_gradient(mean_loc)
            mean_locs.append(mean_loc)
            sample_loc = tf.maximum(-1.0, tf.minimum(1.0, mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)))
        
            # don't propagate throught the locations
            sample_loc = tf.stop_gradient(sample_loc)
            sampled_locs.append(sample_loc)
            
            return sample_loc


def rnn(initial_glimpse_tensor):
    # Scope: CoreNetwork
    inputs = [0]*time_steps
    outputs = [0]*time_steps
    # Scope: GlimpseNetwork
    glimpse_tensor = initial_glimpse_tensor
    hiddenState = None
    REUSE = None
    # Scope: CoreNetwork
    for t in range(time_steps):
        if t == 0:
            hiddenState_prev = tf.zeros((batch_size, cell_size))
        else:
            hiddenState_prev = outputs[t-1]
        
        #calculate the new hidden state
        with tf.variable_scope(c_n_s,auxiliary_name_scope=False,reuse=REUSE) as s1:
            with tf.name_scope(s1.original_name_scope):
                hiddenState = tf.nn.relu(affineTransform(hiddenState_prev, cell_size) + (tf.matmul(glimpse_tensor, Wc_g_h) + Bc_g_h))
        
        inputs[t] = glimpse_tensor
        outputs[t] = hiddenState        
        
        core_net_out = tf.stop_gradient(hiddenState)
        baseline_network(core_net_out)
        
        sample_loc = loc_network(core_net_out)
        g = glimpse_sensor(inputs_placeholder,sample_loc)
        glimpse_tensor = glimpse_network(g,sample_loc)
        
        REUSE = True
    
    core_net_out = tf.stop_gradient(hiddenState)
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))
    
    baseline_network(core_net_out)
    
    return outputs

def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)

def action_network(outputs):
    with tf.variable_scope(a_n_s,auxiliary_name_scope=False) as s1:
        with tf.name_scope(s1.original_name_scope):
            p_y = tf.nn.softmax(tf.matmul(outputs, Wa_h_a) + Ba_h_a)
            return p_y

def get_reward(p_y):
    with tf.variable_scope("Rewards"):
        max_p_y = tf.arg_max(p_y, 1)
        correct_y = tf.cast(labels_placeholder, tf.int64)
        # reward for all examples in the batch
        R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
        reward = tf.reduce_mean(R) # mean reward
        R = tf.reshape(R, (batch_size, 1))
        R = tf.tile(R, [1, (time_steps+1)*2])
    return R,reward

def reinforce(p_y,R,m_lcs,s_lcs):
    with tf.variable_scope(b_n_s,auxiliary_name_scope=False) as s1:
        with tf.name_scope(s1.original_name_scope):
            b = tf.stack(baselines)
            b = tf.concat(axis=2, values=[b, b])
            b = tf.reshape(b, (batch_size, (time_steps+1) * 2))

    no_grad_b = tf.stop_gradient(b)
    
    with tf.variable_scope("p_locs"):
        p_loc = gaussian_pdf(m_lcs, s_lcs)
        p_loc = tf.reshape(p_loc, (batch_size, (time_steps+1) * 2))

    with tf.variable_scope("REINFORCE"):
        J = tf.concat(axis=1, values=[tf.log(p_y + SMALL_NUM) * (onehot_labels_placeholder), tf.log(p_loc + SMALL_NUM) * (R - no_grad_b)])
        J = tf.reduce_sum(J, 1)
        J = J - tf.reduce_sum(tf.square(R - b), 1)
        J = tf.reduce_mean(J, 0)
        cost = -J
    
    var_list = tf.trainable_variables()
    with tf.variable_scope("Grads"):
        grads = tf.gradients(cost, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(lr,name="Optimizer")
        train_op = optimizer.apply_gradients(zip(grads, var_list), global_step=global_step)

    return cost,train_op

def conv_single_tensor():
    with tf.name_scope(sl):
        sam_locs = tf.concat(axis=0, values=sampled_locs)
        sam_locs = tf.reshape(sam_locs, (time_steps+1, batch_size, 2))
        sam_locs = tf.transpose(sam_locs, [1, 0, 2])
    with tf.name_scope(ml):
        me_locs = tf.concat(axis=0, values=mean_locs)
        me_locs = tf.reshape(me_locs, (time_steps+1, batch_size, 2))
        me_locs = tf.transpose(me_locs, [1, 0, 2])
    return sam_locs,me_locs

def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    return tf.Variable(initial, name=myname, trainable=train)

def affineTransform(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def evaluate():
    data = dataset.test
    batches_in__epoch = len(data._images) // batch_size
    accuracy = 0
    
    for i in range(batches_in__epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        nextX, _ = convertTranslated(nextX, MNIST_SIZE, img_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                     onehot_labels_placeholder: dense_to_one_hot(nextY)}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in__epoch
    print(("ACCURACY: " + str(accuracy)))



with tf.Graph().as_default():
            
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)
    
    with tf.variable_scope("Labels"):
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels_raw")
        onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10), name="labels_onehot")
    
    with tf.variable_scope("InputImages"):
        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_size * img_size), name="images")
    
    with tf.name_scope("InitialInputLocation"):
        initial_mean_loc = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)
        initial_sampled_loc = tf.tanh(initial_mean_loc + tf.random_normal(initial_mean_loc.get_shape(), 0, 0.22)) 
    
    with tf.name_scope("MeanLocations") as ml:
        mean_locs.append(initial_mean_loc)   
        
    with tf.name_scope("SampledLocations") as sl:
        sampled_locs.append(initial_sampled_loc)

    with tf.variable_scope("GlimpseNetwork") as g_n_s:
        Wg_l_h = weight_variable((2, hl_size), "glimpseNet_wts_location_hidden", True)
        Bg_l_h = weight_variable((1,hl_size), "glimpseNet_bias_location_hidden", True)
        
        Wg_g_h = weight_variable((total_sensor_bandwidth, hg_size), "glimpseNet_wts_glimpse_hidden", True)
        Bg_g_h = weight_variable((1,hg_size), "glimpseNet_bias_glimpse_hidden", True)
        
        Wg_hg_gf1 = weight_variable((hg_size, g_size), "glimpseNet_wts_hiddenGlimpse_glimpseFeature1", True)
        Wg_hl_gf1 = weight_variable((hl_size, g_size), "glimpseNet_wts_hiddenLocation_glimpseFeature1", True)
        Bg_hlhg_gf1 = weight_variable((1,g_size), "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1", True)
    
    with tf.variable_scope("CoreNetwork") as c_n_s:
        Wc_g_h = weight_variable((cell_size, g_size), "coreNet_wts_glimpse_hidden", True)
        Bc_g_h = weight_variable((1,g_size), "coreNet_bias_glimpse_hidden", True)
    
    with tf.variable_scope("BaselineNetwork") as b_n_s:
        Wb_h_b = weight_variable((g_size, 1), "baselineNet_wts_hiddenState_baseline", True)
        Bb_h_b = weight_variable((1,1), "baselineNet_bias_hiddenState_baseline", True)
    
    with tf.variable_scope("LocationNetwork") as l_n_s:
        Wl_h_l = weight_variable((cell_out_size, 2), "locationNet_wts_hidden_location", True)
        Bl_h_l = weight_variable((1, 2), "locationNet_bias_hidden_location", True)
     
    with tf.variable_scope("ActionNetwork") as a_n_s:
        Wa_h_a = weight_variable((cell_out_size, n_classes), "actionNet_wts_hidden_action", True)
        Ba_h_a = weight_variable((1,n_classes),  "actionNet_bias_hidden_action", True)
    
    
    g = glimpse_sensor(inputs_placeholder,initial_sampled_loc)
    g1 = glimpse_network(g,initial_sampled_loc)
    outputs = rnn(g1)
    sm_locs,m_locs = conv_single_tensor()
    p_y = action_network(outputs)
    R,reward = get_reward(p_y)
    cst,optz = reinforce(p_y,R,sm_locs,m_locs)
    
    #####################################Model starts here####################################  
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('D:\\Shreyas\\RAM-master\\summary\\train', sess.graph)
    for epoch in range(1, max_iters):
        nextX, nextY = dataset.train.next_batch(batch_size)
        nextX, nextX_coord = convertTranslated(nextX, 28, img_size)
        feed_dict = {inputs_placeholder: nextX,labels_placeholder: nextY,onehot_labels_placeholder: dense_to_one_hot(nextY)}
        fetches = [cst,optz]
        cst_fetched,_ = sess.run(fetches,feed_dict=feed_dict)
    
        print("Epoch: ",epoch)
        
        if epoch%5000 == 0:
            evaluate()    
    
    sess.close()
    