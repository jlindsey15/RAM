# A tensorflow implementation of the recurrent attention model


## Intro to RAM

This is an implementation of the RAM (recurrent attention model) model [1], and it is a debugged version of [2]. Instead of using all pixels of the image, this model extracts a local glimpse of the entire image at each time step, and integrate information over time to form a complete representation. Once the model learned where to look, it is more robust against the influence of translation than standard ConvNet, demonstrated by [1].

**For a more detailed description, please refer to the repo [wiki page] (https://github.com/QihongL/RAM/wiki)!**

## Run the RAM

To run the code, simply type `python ram.py [simulation name]` (say, in the terminal). The **model parameters** are described [here] (https://github.com/QihongL/RAM/wiki/Parameter-description) in our RAM wiki page. The input argument `simulation name` will be used to create folders to save the summary log file and images plotting the model's policy (I haven't finish this part yet...). 

It should run if the directory structure is correctly specified. For example, there should be two folders called "summary" and "chckPts" in the project directory.  


## Some results

The `ram.py` implements the RAM. For the 60 X 60 translated MNIST,  it can get 90% accuracy in 2 hours, with a somewhat arbitary choice of parameters. Here're the reward and cost over time: 

<img src="https://github.com/QihongL/RAM/blob/master/demo/rwd_tMnist.png" width="200">
<img src="https://github.com/QihongL/RAM/blob/master/demo/cost_tMnist.png" width="200">


If you found any error in the code, please let us know. Thanks! 

## Prerequisites

Python 2.7 or Python 3.3+

Tensorflow

NumPy

Matplotlib


## References: 

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

[2] https://github.com/seann999/tensorflow_mnist_ram

