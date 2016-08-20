import scipy.io
import os
import numpy as np
import sys

def getData(batchSize, dataSetName,img_size1,img_size2, hasLabel, maxNumObj):

    # parameters
    datapath = '/Users/Qihong/Dropbox/github/mathCognition_RAM/datasets/'
    format = '.mat'
    imgName = 'vectorImg'

    fullDatapath = datapath + dataSetName

    # count how many training images we have in the directory
    files_mat = [i for i in os.listdir(fullDatapath) if i.endswith(format)]
    numExamples = len(files_mat)

    # take a random batch from the training images
    egIdx = np.random.choice(numExamples, batchSize, replace=False) + 1
    # put the all images into a image batch, rep. by a matrix
    imgBatch = np.zeros([batchSize, img_size1*img_size2])
    labels = np.empty([batchSize,])
    coordinates = []
    for i in xrange(batchSize):
        # get one image
        fname = dataSetName + ("%.3d" % egIdx[i]) + format
        file = scipy.io.loadmat(fullDatapath + '/' + fname)

        img = file[imgName]
        # unroll the image to a vector
        imgBatch[i,:] = np.reshape(img[0:img_size1*img_size2],[img_size1*img_size2,])
        # get the image label
        if hasLabel:
            labels[i] = img[img_size1*img_size2]
        # get the coordinates
        curCoords = np.reshape(img[img_size1*img_size2+1:],[maxNumObj,2], 'F')
        coordinates.append(curCoords)

        # testing
        # if i == 0:
        #     print img[img_size1 * img_size2 + 1:], [maxNumObj, 2]


    return imgBatch, labels.astype(int), coordinates, egIdx