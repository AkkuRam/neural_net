# neural_net

A basic neural network was implemented that can perform forward and backpropagation. The aim was to predict the XOR operator. Since we are trying 
to predict an XOR our input is going to be a [1 x 2] vector (i.e. [0, 0], [0, 1], etc.) 

The network gives out a single output, which corresponds to the [1x2] input given. Therefore the target output for each [1x2] vector should look like the 
following:  
[0,0] = 0  
[0,1] = 1  
[1,0] = 1  
[1,1] = 0  

Before training the network gives the following output:

[0,0] = 0.4318125958126001  
[0,0] = 0.4618812892909114  
[0,0] = 0.4306205201741023  
[0,0] = 0.46349028491687544  

After training the network, it approaches the expected target values as seen below:

[0,0] = 0.0068651423091961 (close to 0)  
[0,0] = 0.9137907231894812 (close to 1)   
[0,0] = 0.9126004464154855 (close to 1)   
[0,0] = 0.1082647074600035 (close to 0)   
