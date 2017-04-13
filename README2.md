## Tensorflow Implementation of Recurrent Attention Model (RAM)


## Author

Juntae, Kim, jtkim@kaist.ac.kr

## Requirement

tensorflow rc 1.1.0-rc0

## Description

code: 'ram_modified.py'

This project is modified version of https://github.com/jlindsey15/RAM.
The critical problem of last implemetnation is that the location network cannot learn because of tf.stop_gradient implementation. 
If 'tf.stop_gradient' was commented, the classification result was very bad.
The reason I think is that the problem is originated from sharing the gradient flow through location, core, glimpse network.
Through gradient sharing, gradients of classification part are corrupted by gradients of reinforcement part so that classification result 
become very bad. (If someone want to gradient sharing, the weighted loss should be needed. please refer https://arxiv.org/pdf/1412.7755.pdf)
According to their post research, 'Multiple Object Recognition with Visual Attention' (https://arxiv.org/pdf/1412.7755.pdf) they 
softly separate location network and others through multi-layer RNN. From this, I assume that sharing the gradient through whole network 
is not a good idea so separate them, and finally got a good result. 
In summary, the learning stretegy is as follow. 

1. location network, baseline network : learn through only gradients of reinforcement learning.

2. glimpse network, core network : learn through only gradients of supervised learning.

Thank you!

## Result

After 300,000 epoch, I got about 97% accuracy.

## Reference

Recurrent Models of Visual Attention

http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

https://arxiv.org/pdf/1412.7755.pdf
