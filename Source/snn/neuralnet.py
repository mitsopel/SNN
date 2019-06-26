import snn.math

# System
import sys

# Mathematics
import random
import numpy as np

class NeuralNet:

    def __init__(self, Structure, Activation=snn.math.sigmoid):

        # Set appropriate derivative based on the given
        # function pointer
        self.HiddenActivation = Activation
        if ( Activation == snn.math.sigmoid ):
            self.HiddenActivationDerivative = snn.math.sigmoid_derivative
        elif ( Activation == snn.math.relu ):
            self.HiddenActivationDerivative = snn.math.relu_derivative
        elif ( Activation == snn.math.softplus ):
            self.HiddenActivationDerivative = snn.math.softplus_derivative
        elif ( Activation == snn.math.cos ):
            self.HiddenActivationDerivative = snn.math.cos_derivative
        elif ( Activation == snn.math.tanh ):
            self.HiddenActivationDerivative = snn.math.tanh_derivative
        else:
            self.HiddenActivationDerivative = snn.math.sigmoid_derivative

        # Holds NumberOfLayers
        self.NumberOfLayers = len(Structure)

        # Holds Total Error
        self.TotalError = 0.0

        # Initialize the weight matrices, with the Gaussian N(0,1)
        self.Weights = [ np.random.normal(0,1,(Structure[i], Structure[i+1])) \
                          for i in range(self.NumberOfLayers-1) ]
        # Initialize the biases for all the layers except for the input layer
        self.Biases  = [ np.random.normal(0,1,(Structure[i])) \
                          for i in range(1,self.NumberOfLayers) ]

    def backpropagate( self, Targets ):
        self.Deltas = {}
        self.DErrorDBias = {}
        self.DErrorDWeight = {}

        # Delta in the final output
        # As defined in the project report
        self.Deltas[self.NumberOfLayers-1] = snn.math.cross_entropy_delta( self.LayerOutput[self.NumberOfLayers-1], Targets )

        # Compute the delta's for the other layers
        # Same operation for all subsequent layers
        for LayerIndex in np.arange(self.NumberOfLayers-2, -1, -1):
            self.Deltas[LayerIndex] = np.dot( self.Deltas[LayerIndex+1],  self.Weights[LayerIndex].T ) * self.HiddenActivationDerivative( np.array(self.LayerInput[LayerIndex]) )

        # Calculate final derivative that will be used as the step in gradient ascent
        for LayerIndex in np.arange(self.NumberOfLayers-1, 0, -1):
            self.DErrorDBias[LayerIndex]   = self.Deltas[LayerIndex]
            self.DErrorDWeight[LayerIndex] = np.dot( self.LayerOutput[LayerIndex-1].T, self.Deltas[LayerIndex] )

        # Return update steps as tuple
        return self.DErrorDBias, self.DErrorDWeight

    def train_mini_batch( self, Data, LearningRate, LambdaNormalization ):

        # Split the data into input and output
        Inputs  = [ Entry[0] for Entry in Data ]
        Targets = [ Entry[1] for Entry in Data ]

        # Feed the input through the network
        ForwardResult = self.feedforward( Inputs )


        # Calculate the Cost function including normalization of the weights
        for Target, Out, in zip(Targets, ForwardResult):
            self.TotalError += np.sum(Target * np.log(Out) - LambdaNormalization * 0.5 * np.sum(np.power(self.Weights[1], 2)) - LambdaNormalization * 0.5 * np.sum(np.power(self.Weights[0], 2)))

        # Propagate the error backwards
        self.backpropagate( Targets )

        # Update step is scaled according to the size of the batch
        ScaleFactor = LearningRate / len(Targets)

        # Update the weights and biases using Gradient Ascent
        for layer in np.arange(1,self.NumberOfLayers):
            self.Biases[layer-1]  += LearningRate * np.mean(self.DErrorDBias[layer], axis=0)
            self.Weights[layer-1] += ScaleFactor * self.DErrorDWeight[layer] - LearningRate * LambdaNormalization * self.Weights[layer-1]

    def train( self, TrainData, Epochs, MiniBatchSize, LearningRate, LambdaNormalization, TestData = None ):
        """ Train the network using the stochastic gradient ascent method. """

        for Epoch in np.arange(Epochs):

            # Randomly shuffle data to further help
            # stochastic gradient ascent
            np.random.shuffle(TrainData)

            # Create slices corresponding to each batch
            Batches = [ TrainData[Start : Start + MiniBatchSize] for Start in np.arange(0, len(TrainData), MiniBatchSize) ]

            for Batch in Batches:
                self.train_mini_batch( Batch, LearningRate, LambdaNormalization )

            print("Cross-Entropy Cost : ", self.TotalError / (len(Batches) * MiniBatchSize ))

            if TestData != None:
                Classification = self.classify(TestData)
                Total = len(TestData)
                print ("Epoch {0}: Correctly classified {1} of {2} !!".format(Epoch, Classification, Total))

                # Force print
                sys.stdout.flush()

    def feedforward( self, InputData ):
        # Store inputs and outputs for each of the layers
        self.LayerInput = {}
        self.LayerOutput = {}

        # Layer 1
        # For the input layer, we don't use any activation function
        self.LayerInput[0] = InputData
        self.LayerOutput[0] = np.array(InputData)

        # Layer 2
        # Feed input through the hidden layer
        # and calculate the weighted sum
        self.LayerInput[1] = np.dot( self.LayerOutput[0], self.Weights[0] ) + self.Biases[0]
        # Weighted sum is passed through the selected Activation Function
        self.LayerOutput[1] = np.array( self.HiddenActivation( self.LayerInput[1] ) )

        # Layer 3
        # Input is the weighted sum of the previous layer
        self.LayerInput[2] = np.dot( self.LayerOutput[1], self.Weights[1] ) + self.Biases[1]
        # Final output is squashed through the softmax function in order
        # to get probabilities and ensure Sum(Out) = 1
        self.LayerOutput[2] = np.array([ snn.math.softmax(Vector) for Vector in self.LayerInput[2] ])

        # Return output from last layer
        return self.LayerOutput[2]

    def classify(self, TestData):
        """ Evaluate performance by counting how many examples in Test Dataset are correctly
            evaluated. """
        Correct = 0
        for Test in TestData:
            HotIndex = np.argmax( Test[1] )

            Wrapper2D = [Test[0]]
            #print(Wrapper2D)
            PredictIndex = np.argmax( self.feedforward( Wrapper2D ) )
            Correct = Correct + 1 if HotIndex == PredictIndex else Correct
        return Correct
