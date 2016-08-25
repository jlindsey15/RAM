# The Recurrent attention model in tensorflow 

## Intro to RAM

This is an implementation of the RAM (recurrent attention model) model [1], and it is a debugged version of [2]. Instead of using all pixels of the image, this model extracts a local glimpse of the entire image at each time step, and integrate information over time to form a complete representation. Once the model learned where to look, it is more robust against the influence of translation than standard ConvNet, demonstrated by [1].

## Some results

The `ram_recur.py` implements the RAM. For the 60 X 60 translated MNIST,  it can get 90% accuracy in 2 hours, with a somewhat arbitary choice of parameters. Here're the reward and cost over time: 

<img src="https://github.com/QihongL/RAM/blob/master/demo/rwd_tMnist.png" width="200">
<img src="https://github.com/QihongL/RAM/blob/master/demo/cost_tMnist.png" width="200">


To run the code, simply type `python ram_recur.py` (say, in the terminal). It should works if the directory structure is correctly specified. 

If you found any error in the code, please let us know. Thanks! 

## Prerequisites

Python 2.7 or Python 3.3+

Tensorflow

NumPy

Matplotlib


## References: 

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

[2] https://github.com/seann999/tensorflow_mnist_ram

