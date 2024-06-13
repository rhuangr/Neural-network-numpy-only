# Neural Network from scratch
This project builds a neural network from scratch using exclusively numpy.
The goal of this project was to help me understand the underlying structure of neural networks and the math behind it.
For this reason, this code will be be filled with comments explaining each step :sad:
## Installation
Clone the repository
Make sure numpy is installed
Run the file. While in the cloned project's directory, paste this command into the terminal `python NeuralNetwork.py` 👍
## Notes
To test the neural network, I gave it a **very** simple task of classifying integers from 0 to 9 as even or odd. I decided on this task, because I could easily generate the data needed through code only. Since my goal was to understand neural networks, this task was well enough. 

An input neuron will have an activation value of 0 or 1, depending on if it corresponds to the number trying to be classified. The neural network will then activate the first neuron is the number is odd, or the second neuron if the number is even.

The code snippet below creates a neural network with 10 input neurons, 4 neurons in a middle layer, and 2 output neurons.
Given my example, each input neuron represents a number from 0 to 9. The agent is then trained on 20000 manually generated samples. 

```
nn  =  NeuralNetwork([10,4,2])
nn.train(20000, 0.01)

testBatch  =  createTestBatch(50)
nn.predict(testBatch)

```
Since this code supports any amount of layers and neurons. This code should work with task compatible with supervised deep learning. Feel free to change the the size of the neural network by changing the input argument of the NeuralNetwork constructor. If you do so, the functions dedicated to classify numbers will no longer work, so remember to
