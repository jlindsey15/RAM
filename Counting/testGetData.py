from matplotlib import pyplot
import matplotlib as mpl
import numpy as np
from getData import getData

pyplot.close("all")

showImage = 1
# datasetName = 'oneObj_big'
datasetName = 'multiObj_balanced'
hasLabel = True
img_size1 = 90
img_size2 = 90
batch_size = 10
maxNumObj = 7

selectedImgIdx = 0

imgBatch, nextY, coords, _ = getData(batch_size, datasetName, img_size1, img_size2, hasLabel, maxNumObj)
for i in xrange(batch_size):
    nextY[i] = int(nextY[i])

print 'image batch dimension:'
print np.shape(imgBatch)


thiscoord = coords[selectedImgIdx]
print np.shape(coords)
print np.shape(thiscoord)
print thiscoord[:,0]
print thiscoord[:,1]


print nextY

print np.zeros((batch_size, maxNumObj), dtype=bool)

print np.zeros(batch_size)

# show the 1st image

numObjs = nextY[selectedImgIdx]
print 'number of objects = %d' % (numObjs)

if showImage:
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['black', 'white'])
    tempImg = np.reshape(imgBatch[selectedImgIdx,:], [img_size1, img_size2])
    # tempImg = np.transpose(tempImg)
    fig = pyplot.imshow(tempImg, interpolation='nearest', cmap=cmap)
    pyplot.colorbar(fig, cmap=cmap, ticks=[1, 0])
    pyplot.show()