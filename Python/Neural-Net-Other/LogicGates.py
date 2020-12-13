from NeuralNetwork.Network import *
from NeuralNetwork.Tools import make_error_graph
from NeuralNetwork.Functions import *
import numpy as np


# DEBUG = True
DEBUG = False
# SUMMARY_FREQ = 1
SUMMARY_FREQ = 100
# N_EPOCHS = 10
N_EPOCHS = 10000
#
TRAIN = True
# TRAIN = False

if __name__ == '__main__':
    data_sets = {"XOR":  np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]]),
                 "OR":   np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]),
                 "AND":  np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]),
                 "XNOR": np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])}

    gate = "XOR"

    data = data_sets[gate]

    np.random.shuffle(data)    # TODO: maybe try shuffling the data in the epoch loop

    inputs = data[:, 0:2]
    labels = data[:, -1]

    val_inputs = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    val_outputs = np.array([1, 0, 0, 1])

    ########################################
    network = Network(input_size=len(inputs[0]),
                      output_size=2,
                      learning_rate=0.05)
    # hidden layers:
    network.add_layer(Dense(size=6,
                            activation_fn=sigmoid
                            ))


    # output:
    network.add_layer(Dense(size=network.output_size, activation_fn=sigmoid))
    ########################################

    if DEBUG:
        print(network)

    if TRAIN:
        error_data, validation_data = network.train(inputs, labels,
                                                    N_EPOCHS, SUMMARY_FREQ, DEBUG,
                                                    val_inputs, val_outputs)
        make_error_graph(error_data, validation_data)
        # save weights:
        network.save_weights("Weights\\XOR.npy")
        print("Saved weights.")
    else:
        network.load_weights("Weights\\XOR.npy")
        print("Loaded weights.")

    while True:
        print("q to quit")
        user_inputs = ["0"] * len(inputs[0])
        for input_index in range(len(inputs[0])):
            user_inputs[input_index] = input("Input_" + str(input_index) + ":")
            user_inputs = ["0" if x == "" else x for x in user_inputs]
            if user_inputs[input_index].lower().startswith('q'):
                quit()

        user_ints = [1 if (user_inputs[ind].lower().startswith('t') or user_inputs[ind].lower().startswith('1'))
                     else 0 for ind in range(len(user_inputs))]

        network_output = np.argmax(network.predict(np.array(user_ints).astype(int)))
        if DEBUG:
            print(network)
        print("=> Output:", network_output)
        output_string = "True" if network_output == 1 else "False"
        print("\t", user_inputs[0], gate, user_inputs[1], "=>", output_string)

        print("_"*100)
