import tensorflow as tf
import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import time
import math

dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "save-3scales/"
save_prefix = "save"
start_step = 10000
#load_path = None
load_path = save_dir + save_prefix + str(start_step) + ".ckpt"
# to enable visualization, set draw to True
eval_only = False
animate = True
draw = True

# model parameters
minRadius = 4               # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8         # fixed resolution of sensor
depth = 3                  # number of zooms
channels = 1                # mnist are grayscale images
totalSensorBandwidth = depth * channels * (sensorBandwidth **2)

# number of units
hg_size = 128               # glimpse
hl_size = 128               # location
g_size = 256                #
cell_size = 256             #
cell_out_size = cell_size   #

glimpses = 7                # number of glimpses
n_classes = 10              # cardinality(Y)

batch_size = 10
max_iters = 1000000

mnist_size = 28             # side length of the picture

loc_sd = 0.1                # std when setting the location
mean_locs = []              #
sampled_locs = []           # ~N(mean_locs[.], loc_sd)
glimpse_images = []         # to show in window

# set the weights to be small random values, with truncated normal distribution
def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    return tf.Variable(initial)

# get local glimpses
def glimpseSensor(img, normLoc):
    loc = ((normLoc + 1) / 2) * mnist_size # normLoc coordinates are between -1 and 1
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (batch_size, mnist_size, mnist_size, channels))

    zooms = []
    
    # process each image individually
    for k in xrange(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        max_radius = minRadius * (2 ** (depth - 1)) 
        offset = 2 * max_radius
    
        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
            max_radius * 4 + mnist_size, max_radius * 4 + mnist_size)
        
        for i in xrange(depth):
            r = int(minRadius * (2 ** (i - 1)))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])

            d = tf.tile(d, [2])
            
            loc_k = loc[k,:]
            print(loc_k.get_shape())
            adjusted_loc = offset + loc_k - r
            
            
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,\
                one_img.get_shape()[1].value))
                
            # crop image to (d x d)

            print(d_raw)
            zoom = tf.slice(one_img2, adjusted_loc, d)
            
            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)
    
        zooms.append(tf.pack(imgZooms))
        
    zooms = tf.pack(zooms)
    
    glimpse_images.append(zooms)
    
    return zooms

# implements the glimpse network
def get_glimpse(loc):
    # get glimpse using the previous location
    glimpse_input = glimpseSensor(inputs_placeholder, loc)
    glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))

    # the hidden units that process location & the glimpse
    l_hl = weight_variable((2, hl_size))
    glimpse_hg = weight_variable((totalSensorBandwidth, hg_size))
    hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
    hl = tf.nn.relu(tf.matmul(loc, l_hl))

    # the hidden units that integrates the location & the glimpses
    hg_g = weight_variable((hg_size, g_size))
    hl_g = weight_variable((hl_size, g_size))
    g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))   # TODO linear layer in Mnih et al. (2014)!
    g2 = tf.matmul(g, intrag)
    return g


def get_next_input(output, i):
    # the next location is computed by the location network
    mean_loc = tf.tanh(tf.matmul(output, h_l_out))
    mean_locs.append(mean_loc)
    
    sample_loc = tf.tanh(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd))
    sampled_locs.append(sample_loc)
    
    return get_glimpse(sample_loc)


def model():
    # initialize the location under unif[-1,1], for all example in the batch
    initial_loc = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)
    # get the glimpse using the glimpse network
    initial_glimpse = get_glimpse(initial_loc)   

    #
    # lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple = True, num_proj=cell_out_size)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    #
    inputs = [initial_glimpse]
    inputs.extend([0] * (glimpses - 1))

    #
    outputs, _ = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=get_next_input)
    # get the next location
    get_next_input(outputs[-1], 0)
            
    return outputs


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  # copied from TensorFlow tutorial
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * n_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


# to use for maximum likelihood with glimpse location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)


def calc_reward(outputs):
    # conside the action at the last time step
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))
    b = tf.sigmoid(tf.matmul(outputs, b_weights))
    # the hidden layer for the action network
    h_a_out = weight_variable((cell_out_size, n_classes))
    # process its output
    p_y = tf.nn.softmax(tf.matmul(outputs, h_a_out))
    max_p_y = tf.arg_max(p_y, 1)
    # the targets
    correct_y = tf.cast(labels_placeholder, tf.int64)

    # reward for all examples in the batch
    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
    reward = tf.reduce_mean(R) # mean reward

    #
    p_loc = gaussian_pdf(mean_locs, sampled_locs)
    p_loc = tf.reshape(p_loc, (batch_size, glimpses * 2))

    print(R)
    R = tf.reshape(R, (batch_size, 1))
    print(R)
    # 1 means concatenate along the row direction
    J = tf.concat(1, [tf.log(p_y + 1e-5) * onehot_labels_placeholder, tf.log(p_loc + 1e-5) * (R - b_placeholder)])
    print(J)
    # sum the probability of action and location
    J = tf.reduce_sum(J, 1)
    print(J)
    # average over batch
    J = tf.reduce_mean(J, 0)
    print(J)
    cost = -J
    cost = cost + tf.square(tf.reduce_mean(R - b))

    # Adaptive Moment Estimation
    # estimate the 1st and the 2nd moment of the gradients
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(1e-3, global_step, 1000, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(cost)

    return cost, reward, max_p_y, correct_y, train_op, b, tf.reduce_mean(b), tf.reduce_mean(R - b)


def evaluate():
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0

    for i in xrange(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                     onehot_labels_placeholder: dense_to_one_hot(nextY)}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))




with tf.Graph().as_default():
    # the y vector
    labels = tf.placeholder("float32", shape=[batch_size, n_classes])
    # the input x and yhat
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28 * 28), name="images")
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels")
    onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10), name="oneHotLabels")
    b_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1), name="b")


    #
    h_l_out = tf.ones((cell_out_size, 2))
    loc_mean = weight_variable((batch_size, glimpses, 2))
    intrag = weight_variable((g_size, g_size))
    b_weights = weight_variable((g_size, 1))

    # query the model ouput
    outputs = model()
    
    # convert list of tensors to one big tensor
    sampled_locs = tf.concat(0, sampled_locs)
    sampled_locs = tf.reshape(sampled_locs, (batch_size, glimpses, 2))
    mean_locs = tf.concat(0, mean_locs)
    mean_locs = tf.reshape(mean_locs, (batch_size, glimpses, 2))
    glimpse_images = tf.concat(0, glimpse_images)

    #
    cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb = calc_reward(outputs)

    tf.scalar_summary("reward", reward)
    tf.scalar_summary("cost", cost)
    summary_op = tf.merge_all_summaries()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    b_fetched = np.zeros((batch_size, 1))
    
    # ckpt = tf.train.get_checkpoint_state(save_dir)
    # if load_path is not None and ckpt and ckpt.model_checkpoint_path:
    #     try:
    #         saver.restore(sess, load_path)
    #         print("LOADED CHECKPOINT")
    #     except:
    #         print("FAILED TO LOAD CHECKPOINT")
    #         exit()
    # else:
    init = tf.initialize_all_variables()
    sess.run(init)

    if eval_only:        
        evaluate()
    else:
        summary_writer = tf.train.SummaryWriter("summary", graph=sess.graph)
        
        if draw:
            fig = plt.figure()
            txt = fig.suptitle("-", fontsize=36, fontweight='bold') 
            plt.ion()
            plt.show()   
            plt.subplots_adjust(top=0.7)
            plotImgs = []

        # training
        for step in xrange(start_step + 1, max_iters):
            start_time = time.time()

            # get the next batch of examples
            nextX, nextY = dataset.train.next_batch(batch_size)
            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, onehot_labels_placeholder: dense_to_one_hot(nextY), b_placeholder: b_fetched}
            fetches = [train_op, cost, reward, predicted_labels, correct_labels, glimpse_images, b, avg_b, rminusb]
            # feed them to the model
            results = sess.run(fetches, feed_dict=feed_dict)
            _, cost_fetched, reward_fetched, prediction_labels_fetched,\
                correct_labels_fetched, f_glimpse_images_fetched, b_fetched, avg_b_fetched, rminusb_fetched = results
            
            duration = time.time() - start_time
            
            if step % 20 == 0:
                if step % 1000 == 0:
                    saver.save(sess, save_dir + save_prefix + str(step) + ".ckpt")
                    if step % 5000 == 0:
                        evaluate()

    
                ##### DRAW WINDOW ################
    
                f_glimpse_images = np.reshape(f_glimpse_images_fetched, (glimpses + 1, batch_size, depth, sensorBandwidth, sensorBandwidth)) #steps, THEN batch
                
                if draw:
                    if animate:
                        fillList = False
                        if len(plotImgs) == 0:
                            fillList = True
                        
                        # display first in mini-batch
                        for y in xrange(glimpses):
                            txt.set_text('FINAL PREDICTION: %i\nTRUTH: %i\nSTEP: %i/%i'
                                % (prediction_labels_fetched[0], correct_labels_fetched[0], (y + 1), glimpses))
                            
                            for x in xrange(depth):
                                plt.subplot(depth, 1, x + 1)
                                if fillList:
                                    plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                                         interpolation="nearest")
                                    plotImg.autoscale()                                
                                    plotImgs.append(plotImg)
                                else:
                                    plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                                    plotImgs[x].autoscale()  
                                    
                            fillList = False
                            
                            fig.canvas.draw()
                            time.sleep(0.1)
                            plt.pause(0.0001) 
                    else:
                        txt.set_text('PREDICTION: %i\nTRUTH: %i' % (prediction_labels_fetched[0], correct_labels_fetched[0]))  
                        for x in xrange(depth):
                            for y in xrange(glimpses):
                                plt.subplot(depth, glimpses, x * glimpses + y + 1)
                                plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                           interpolation="nearest")
                        
                        plt.draw()
                        time.sleep(0.05)
                        plt.pause(0.0001)  
                        
                ################################
                
                print('Step %d: cost = %.5f reward = %.5f (%.3f sec) b = %.5f R-b = %.5f' % (step, cost_fetched, reward_fetched, duration, avg_b_fetched, rminusb_fetched))
                
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            
sess.close()
