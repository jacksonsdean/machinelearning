from numpy import exp, array, random, dot, genfromtxt
import csv
import pandas as p

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((784, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

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
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


def get_training_data_input():
    training_data = array([])
    labels = []
    i = 0
    with open('./data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader)
        labels = reader[1]
        for row in reader:
            if i > 1000:
                break
            i += 1
            img_data = []
            for val in row[1:]:
                # print(val)
                f = float(val)
                img_data.append(f)

            training_data.append(img_data)
    print(labels)
    return training_data # the first row is just headers

def get_training_data_output():
    training_data = []
    i = 0
    with open('./data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader)

        for row in reader:
            if i > 1000:
                break
            i += 1
            img_data = []
            for val in row[1:]:
                f = float(val)
                img_data.append(f)
            training_data.append(img_data)
    return training_data # the first row is just labels

def get_test_input():
    i = 0
    test_input = [1]
    with open('./data/test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader)
        for row in reader:
            if i > 1000:
                break
            i += 1
            img_data = []
            for val in row[1:]:
                f = float(val)
                img_data.append(f)
            test_input.append(img_data)
    return test_input # the first row is just labels

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    # print("Random starting synaptic weights: ")
    # print(len(neural_network.synaptic_weights))

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    # training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

    input = array(get_training_data_input())
    output = array(get_training_data_output())
    # for i in inspect.getmembers(input.size):
        # print(i[0])

    print("input size:",input.shape)
    print("output size:",output.shape)

    training_set_inputs = input
    training_set_outputs = output

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10)

    with open("out.txt", "w") as file:
        for w in neural_network.synaptic_weights:
            file.write(str(w) + "\n")
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering test data: ")
    print(neural_network.think(get_test_input()))