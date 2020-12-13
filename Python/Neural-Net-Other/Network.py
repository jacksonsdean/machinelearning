from random import shuffle
from NeuralNetwork.Functions import *
from NeuralNetwork import Tools

# np.random.seed(42)


class Neuron:
    def __init__(self, layer, n_weights=0, output=0):
        self.layer = layer
        self.weights = np.random.uniform(size=n_weights)
        self.output = output
        self.delta = 0

    def __str__(self):
        return "<Neuron: | Result: %.3f | Weights: " % self.output + \
               str(list(self.weights.flat)) + "| Delta: %.3f>" % self.delta

    def feed_forward_neuron(self, last_layer):
        _sum = 0
        for index, last_layer_neuron in enumerate(last_layer.neurons):
            _sum += self.weights[index] * last_layer_neuron.output
        if self.layer.activation_fn == dropout:
            self.output = self.layer.activation_fn(_sum, self.layer.rate) + self.layer.bias
        else:
            self.output = self.layer.activation_fn(_sum) + self.layer.bias


class Layer:
    def __init__(self, network, size, index, activation_fn=sigmoid, bias=0):
        self.network = network
        self.neurons = []
        self.activation_fn = activation_fn
        self.size = size
        self.bias = bias
        self.index=index
        self.change_index(self.index)

    def feed_forward_layer(self, prev_layer):
        for neuron in self.neurons:
            neuron.feed_forward_neuron(prev_layer)

    def get_neurons_string(self):
        output = ""
        for n in self.neurons:
            output += str(n) + ", "
        return output

    def get_neuron_outputs(self):
        output = []
        for n in self.neurons:
            output.append(n.output)
        return np.array(output).reshape(1, len(self.neurons))

    def change_index(self, index):
        self.index = index
        self.neurons = []
        last_layer_size = self.network.get_last_layer_size(self.index)
        for index in range(self.size):
            self.neurons.append(Neuron(self, last_layer_size))

class Dense(Layer):
    def __init__(self, network, size=0, index=0, activation_fn=sigmoid):
        super().__init__(network, size, index, activation_fn)


class Dropout(Layer):
    def __init__(self, network, size=0, index=0, activation_fn=dropout, rate=1):
        super().__init__(network, size, index, activation_fn)
        self.rate = rate


class Network:
    def __init__(self, input_size, output_size, learning_rate=0.05):
        self.initialized = False
        self.debugging = False
        self.is_training = False
        self.stop = False
        self.layers = []
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.epoch = 0
        self.progress_string = "Press Train to begin"
        self.input_layer = None

        self.setup_input_layer(self.input_size)
        # input layer:
        self.set_inputs([0]*input_size)

    def __str__(self):
        string = ""
        string += "Network Structure:" \
                  "\n   |-Input Layer:"
        for neuron in self.input_layer.neurons:
            string += "\n      |---" + str(neuron)
        string += "\n   |-Hidden Layers:"
        for layer in self.layers[:-1]:
            string += "\n   |--Layer%d (" % layer.index +\
                      str(layer.activation_fn.__name__) + "):"
            for neuron in layer.neurons:
                string += "\n      |---" + str(neuron)
        string += "\n   |--Output layer:"
        for neuron in self.layers[-1].neurons:
            string += "\n      |---" + str(neuron)
        return string

    def setup_input_layer(self, size:int):
        self.input_size = size
        self.input_layer = Dense(self, size, index=-1)

    def add_layer(self, new_layer, position=-1):
        self.initialized = True
        # if this is the first hidden layer, last layer is input, else it's the last hidden layer
        last_layer_size = self.input_size if len(self.layers) < 1 else len(self.layers[-1].neurons)
        new_layer.last_layer_size = last_layer_size

        index = len(self.layers)

        # index = 0 if len(self.layers) < 2 else len(self.layers)-1
        if position != -1:
            # insert
            index = position

        new_layer.change_index(index)
        self.layers.insert(index, new_layer)
        # make new layer's neurons
        # for neuron_index in range(new_layer.size):
        #     new_layer.neurons.append(Neuron(new_layer, n_weights=last_layer_size))


    def set_inputs(self, inputs: list):
        self.setup_input_layer(self.input_size)
        # for n in range(len(inputs)):
        #     self.input_layer.neurons.append(Neuron(self, n_weights=0, output=inputs[n]))

    def train(self, training_inputs, training_labels,
              n_epochs: int, summary_freq = 1, debug=False,
              validation_inputs=np.array([]), validation_outputs=np.array([])):

        self.is_training = True
        use_validation = len(validation_inputs) > 0

        print("Training data:\n\t" + str(len(training_inputs)) + " inputs" + "\n\t"
              + str(len(training_labels)) + " labels")
        print("Starting training with", n_epochs, "epochs")

        self.debugging = debug

        if self.debugging:
            print(self)
        sum_error = 0
        val_error = 0
        train_error_data_points = {}
        val_error_data_points = {}
        for self.epoch in range(n_epochs):
            if self.stop:
                self.stop = False
                return train_error_data_points, val_error_data_points
            # shuffle the lists:
            # zipped = list(zip(training_inputs, training_labels))
            # shuffle(zipped)
            # training_inputs, training_labels = zip(*zipped)
            # training_inputs = np.array(training_inputs)
            # training_labels = np.array(training_labels)

            sum_error = 0
            for row_index, row in enumerate(training_inputs):
                # one-hot encoding
                expected = [0.0] * self.output_size
                expected[training_labels[row_index]] = 1
                if self.debugging:
                    print("." * 200)
                    print("Row Input:", row)
                    print("Expected output:", expected)

                self.set_inputs(row)                                # populate first layer:

                outputs = self.feed_forward(self.input_layer)       # feed-forward through the network
                # using sum of errors loss function:

                sum_error = 0                                       # calculate error
                for j in range(len(expected)):
                    sum_error += (expected[j] - outputs[j])**2

                self.back_propagation(expected)                     # back-propagate, calculating deltas
                self.update_weights(row)                            # update all the weights

                if self.debugging:
                    print("Sum of errors:", sum_error)
                    print(self)

            if self.epoch % summary_freq == 0:
                train_error_data_points[self.epoch] = sum_error
                self.progress_string = "Epoch %d / %d | Training Error: %.3f " % (self.epoch, n_epochs, sum_error)

                if use_validation:
                    val_error = self.validate(validation_inputs, validation_outputs)
                    val_error_data_points[self.epoch] = val_error
                    self.progress_string += "| Validation Error: %.3f | " % val_error
                if not self.debugging:
                    Tools.print_progress_bar(self.epoch, n_epochs,
                                             prefix=self.progress_string,
                                             length=30,
                                             print_end="")
                    Tools.make_error_graph(train_error_data_points, val_error_data_points)

        print("\nFinished train")

        self.progress_string = "Finished %d epochs | Training Error: %.3f " % (n_epochs, sum_error)

        if use_validation:
            self.progress_string += "| Validation Error: %.3f | " % val_error

        self.is_training = False

        return train_error_data_points, val_error_data_points

    def validate(self, val_inputs, val_outputs):
        val_error = 0
        for index, input_i in enumerate(val_inputs):
            expected = np.array([0.0] * self.output_size)
            expected[val_outputs[index]] = 1
            self.set_inputs(val_inputs[index])
            actual_outputs = np.array(self.feed_forward(self.input_layer))
            for j in range(len(expected)):
                val_error += (expected[j] - actual_outputs[j]) ** 2
        return val_error

    def feed_forward(self, input_layer):
        # feed input layer to hidden 1:
        prev_layer = input_layer
        for layer in self.layers:           # for all but the first layer
            layer.feed_forward_layer(prev_layer)  # feed the previous layer through this one
            prev_layer = layer              # the next layer will get this layers results

        output = self.layers[-1].get_neuron_outputs()

        if self.debugging:
            print("Actual output:", output)

        return output.flat

    def back_propagation(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = np.array([])
            if i == len(self.layers) - 1:       # if this is the output layer:
                # calculate errors
                results = [neuron.output for neuron in layer.neurons]
                errors = np.array(results) - expected
            else:
                # this is not the last layer
                for j in range(len(layer.neurons)):                     # for hidden layers
                    h_error = 0
                    next_layer = self.layers[i + 1]
                    for neuron in next_layer.neurons:
                        h_error -= (neuron.weights[j] * neuron.delta)
                    errors = np.append(errors, [h_error])

            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.delta = errors[j] * fn_derivatives[layer.activation_fn](neuron.output)

    def update_weights(self, network_input):
        for i in range(len(self.layers)):
            inputs = network_input
            if i != 0:  # if this is not the first layer
                # the inputs are the results of the last layer
                inputs = [neuron.output for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                if len(neuron.weights) == 0:
                    continue
                for j in range(len(inputs)):
                    neuron.weights[j] += self.learning_rate * neuron.delta * inputs[j]

    def predict(self, input_values):
        self.set_inputs(input_values)
        self.feed_forward(self.input_layer)
        return self.layers[-1].get_neuron_outputs()        # output is last layer values

    def save_weights(self, path):
        max_neurons = 0
        for l in self.layers:
            if len(l.neurons) > max_neurons:
                max_neurons = len(l.neurons)

        saved_weights = np.zeros((len(self.layers), max_neurons, max_neurons+1))
        li = 0
        ni = 0
        wi = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    saved_weights[li][ni][wi] = w
                    wi += 1
                ni += 1
                wi = 0
            li += 1
            ni =0
        np.save(path, saved_weights)

    def load_weights(self, path):
        saved_weights = np.load(path)
        li = 0
        ni = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                for wi in range(len(neuron.weights)):
                    neuron.weights[wi] = saved_weights[li][ni][wi]
                ni += 1
            li += 1
            ni = 0
        np.save(path, saved_weights)

    def get_last_layer_size(self, index):
        # TODO: new system where input is layer -1 and 0 is the first layers[] index
        if index < 0:
            return 0
        elif index == 0:
            return self.input_size
        else:
            return len(self.layers[index-1].neurons)

