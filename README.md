Artificial Neural Network Anna
==============================

Use this framework to train and run artificial neural networks. Training is based on gradient descent
algorithm. To achieve best performance, the framework utilizes an OpenCL computing device (e.g. GPU).
The framework supports both single (32-bit) and double (64-bit) floating-point precision for math calculations.
Since a typical GPU has more 32-bit computing units than 64-bit ones, use single precision to achieve 
maximum performance. Use double precision when computational precision is more important than speed
(e.g. a neural network with large number of intermediate layers).
<p>
Here is sample code demonstrating how to use this framework:
```java
// Create a neural network that runs on OpenCL platform 0 and OpenCL device 0, 
// uses 64-bit floating point precision, takes 5 input features, has 3 layers with 4 nodes each
// and does a binary classification (i.e. has 2 labels)      
org.anna.NeuralNetwork nn = new org.anna.NeuralNetwork(0, 0, 
    new org.anna.NetworkDescription(org.anna.FloatingPointPrecision.DOUBLE, 5, 3, 4, 2));

// Train the neural network on the input samples and labels using maximum 100 iterations and no normalization
double[] samples = ... ; // the array size should be 5 (number of features) multiplied by the number of samples
int[] labels = ... ; // the array size should match the number of samples
nn.train(samples, labels, 100, 0.0, null);

// Classify a new sample by running the trained neural network with 0.5 as the classification threshold
double[] newSample = ... ; // the array should contain 5 elements
double[] result = nn.run(newSample);
boolean classification = result[0] >= 0.5;                
```
Reach the author at veugene@yahoo.com
