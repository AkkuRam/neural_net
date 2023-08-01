# neural_net

A basic neural network was implemented that can perform forward and backpropagation. The aim was to predict the XOR operator. Since we are trying 
to predict an XOR our input is going to be a [1 x 2] vector (i.e. [0, 0], [0, 1] etc.) 

The network gives out a single output, which corresponds to the [1x2] input given. Therefore the target output for each [1x2] vector should look like the 
following:
[0,0] = 0\\
[0,1] = 1\\
[1,0] = 1\\
[1,1] = 0\\
