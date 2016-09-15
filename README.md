# A tensorflow implementation of the recurrent attention model


## Intro to RAM

This is an implementation of the RAM (recurrent attention model) described in [1], using some code from the partial implementation found at [2]. Instead of processing all pixels of the image at once, this model focuses on a smaller glimpse window at each time step (the choice of glimpse location is learned). It integrates the information over time to output a classification prediction for the image. Once trained, it is more robust against the influence of translation than a standard ConvNet, demonstrated by [1].

**For a more detailed description, please refer to the repo [wiki page] (https://github.com/QihongL/RAM/wiki)!**

## Run the RAM

To run the code, simply type `python ram.py [simulation name]` (say, in the terminal). The **model parameters** are described [here] (https://github.com/QihongL/RAM/wiki/Parameter-description) in our RAM wiki page. The input argument `simulation name` will be used to create folders to save the summary log file and images plotting the model's policy (I haven't finished this part yet...). 

It should run if the directory structure is correctly specified. For example, there should be two folders called "summary" and "chckPts" in the project directory.  


## Some results

The `ram.py` implements the RAM. For the 60 X 60 translated MNIST,  it converges at 6% error. Here's a comparison between the model with the value baseline prediction term (purple), and the model without the baseline term (blue). The plot shows the reward and cost over time. In this simulations, both model ends up with a similar error (6%), which is something that still needs to be understood...  

<img src="https://github.com/QihongL/RAM/blob/master/demo/b.vs.nb_trans_abs_rwd.png" width="1000">
<img src="https://github.com/QihongL/RAM/blob/master/demo/b.vs.nb_trans_abs_cost.png" width="1000">


If you find any errors in the code, please let us know. Thanks! 

## Prerequisites

Python 2.7 or Python 3.3+

Tensorflow

NumPy

Matplotlib


## References: 

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

[2] https://github.com/seann999/tensorflow_mnist_ram

