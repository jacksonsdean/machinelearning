import numpy as np
import csv
from pandas import read_csv

class NeuralNetwork():
    neurons = 512
    def __init__(self):
        # Seed the np.random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(1)

        # We model a single neuron, with 784 input connections and 10 output connection.
        # We assign np.random weights to a 784 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * np.random.random((784, self.neurons)) - 1
        # self.synaptic_weights_1 = 2 * np.random.random((10, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

    def predict(self, input):
        self.think(input)
        return np.argmax(self.synaptic_weights[-1])



    def get_training_data_input(self):
        training_data = np.zeros((self.neurons, 784))
        csv_file = './data/train.csv'
        dataFrame = read_csv(csv_file, header=0)
        # for each column
        for col in dataFrame.columns:
            # for value in dataFrame[col][1:]:
            np.append(training_data, dataFrame[col])
        return training_data # the first row is just headers

    def get_training_data_output(self):
        labels = np.zeros((self.neurons))
        csv_file = './data/train.csv'
        dataFrame = read_csv(csv_file)
        labels_df = dataFrame.label
        np.append(labels, labels_df)
        return labels # the first row is just labels

    def get_test_data(self):
        result = dict()
        test_case = 500

        one_img_arr = []
        csv_file = './data/train.csv'
        dataFrame = read_csv(csv_file, header=0)

        result["label"] = dataFrame[dataFrame.columns[0]][test_case]
        for col in dataFrame.columns[1:]:
            one_img_arr.append(dataFrame[col][test_case])

        result["arr"] = one_img_arr
        return result

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    n_net = NeuralNetwork()

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    # training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

    input = n_net.get_training_data_input()
    output = n_net.get_training_data_output()

    print("input size:", input.shape)
    print("output size:", output.shape)

    training_set_inputs = input
    training_set_outputs = output

    # Train the neural network using a training set.
    n_net.train(training_set_inputs, training_set_outputs, 10000)

    with open("out.txt", "w") as file:
        for w in n_net.synaptic_weights:
            file.write(str(w) + "\n")

    print("Saving new synaptic weights after training: ")
    np.save(file='results/weights.npy', arr=n_net.synaptic_weights)

    test_data = n_net.get_test_data()

    print("\nTEST PREDICTION:\nActual:",test_data["label"] , "Prediction:", n_net.predict(test_data["arr"]))

    # Test the neural network with a new situation.
    # print("Considering test data: ")
    # print(n_net.think(get_test_input()))