from NeuralNetwork.Network import *
from NeuralNetwork.Tools import make_error_graph
from NeuralNetwork.Functions import *
import pandas as pd
import numpy as np


def code_iris(name, decode=False):
    if not decode:
        if name == 'Iris-setosa':
            return 0
        if name == 'Iris-versicolor':
            return 1
        if name == 'Iris-virginica':
            return 2
    else:
        if name == 0:
            return 'Iris-setosa'
        if name == 1:
            return 'Iris-versicolor'
        if name == 2:
            return 'Iris-virginica'


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.shuffle(df)
    split_1 = int(train_percent * len(df))
    split_2 = int(0.9 * len(df))
    train = df[:split_1]
    validate = df[split_1:split_2]
    test = df[split_2:]
    return train, validate, test


def load_dataset(path:str):
    data = np.array(pd.read_csv(path))

    train, validate, test = train_validate_test_split(data, .8, .2)
    t_inputs = train[:, 1:-1]
    t_labels = train[:, -1]
    t_labels = [code_iris(name, False) for name in t_labels]

    v_inputs = validate[:, 1:-1]
    v_labels = validate[:, -1]
    v_labels = [code_iris(name, False) for name in v_labels]

    return t_inputs, t_labels, v_inputs, v_labels


# DEBUG = True
DEBUG = False
# SUMMARY_FREQ = 1
SUMMARY_FREQ = 1
# N_EPOCHS = 10
N_EPOCHS = 1500

TRAIN = True
# TRAIN = False

if __name__ == '__main__':
    train_inputs, train_labels, validation_inputs, validation_labels = load_dataset("Data\\iris.csv")


    ########################################
    network = Network(input_size=len(train_inputs[0]),
                      output_size=3,
                      learning_rate=0.01)
    # hidden layers:
    # network.add_layer(Dense(size=1000,
    #                         activation_fn=relu
    #                         ))
    # network.add_layer(Dense(size=500,
    #                         activation_fn=relu
    #                         ))
    network.add_layer(Dense(size=64,
                            activation_fn=sigmoid
                            ))
    # network.add_layer(Dropout(size=300,rate=.2))

    # output:
    network.add_layer(Dense(size=network.output_size, activation_fn=sigmoid))
    ########################################
    # print(network)

    if TRAIN:
        error_data, validation_data = network.train(train_inputs, train_labels,
                                                    N_EPOCHS, SUMMARY_FREQ, DEBUG,
                                                    validation_inputs, validation_labels)
        make_error_graph(error_data, validation_data)
        # save weights:
        network.save_weights("Weights\\iris.npy")
        print("Saved weights.")
    else:
        network.load_weights("Weights\\iris.npy")
        print("Loaded weights.")

    while True:
        print("q to quit")
        user_inputs = ["0"] * len(train_inputs[0])
        for input_index in range(len(train_inputs[0])):
            user_inputs[input_index] = input("Input_" + str(input_index) + ":")
            user_inputs = ["0" if x == "" else x for x in user_inputs]
            if user_inputs[input_index].lower().startswith('q'):
                quit()

        if len(user_inputs) == 4:
            network_output = np.argmax(network.predict(np.array(user_inputs).astype(float)))

            output_string = code_iris(network_output, True)
            print("Computer thinks that's an", output_string)
        elif len(user_inputs) == 2:
            user_ints = [1 if user_inputs[ind].lower().startswith('t') else 0 for ind in range(len(user_inputs))]
            network_output = np.argmax(network.predict(np.array(user_ints).astype(int)))

            output_string = "True" if network_output == 1 else "False"
            print("\t", user_inputs[0], "XOR", user_inputs[1], "=>", output_string)

        print("_"*100)
