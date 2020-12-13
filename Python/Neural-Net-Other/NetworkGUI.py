from tkinter import *
from tkinter import Text
from tkinter.ttk import Progressbar, Treeview

from NeuralNetwork.Network import *
from NeuralNetwork.Tools import make_error_graph

DEBUG = True
learning_rate = 0.05

frame_pad = 20

class GUI:
    def __init__(self):
        self.network = None
        self.n_epochs = 0
        self.summary_freq = 0
        self.window = Tk()
        self.window.title("Neural Network")
        self.window.geometry("850x800+650+0")
        self.auto_update_weights = False

        # *********** NETWORK ***********
        self.network_frame = Frame(self.window, padx=10, pady=10, bd=4, relief=RAISED, width=700, height=300)
        self.network_frame.grid(row=0, column=0, sticky=N+W, padx=frame_pad, pady=frame_pad, columnspan=2)

        network_label = Label(self.network_frame, bd=5, font="arial 16 bold", text="Network Structure:")
        network_label.pack(padx=10, pady=10,)

        self.network_tree = Treeview(self.network_frame, height=15)
        self.network_tree.column("#0", minwidth=500, width=500, stretch=YES)
        self.network_tree.pack(fill=BOTH, padx=5, pady=5)

        self.t_auto_update = Button(self.network_frame, text="Auto Update Off", command = self.toggle_auto_update)
        self.t_auto_update.pack()

        Button(self.network_frame, text="Update Weights", command = self.update_network_tree).pack()

        network_controls_frame = Frame(self.window, padx=10, pady=10, bd=4, relief=RAISED, width=200)
        network_controls_frame.grid(row=0, column=2, sticky=N+W, padx=frame_pad, pady=frame_pad)

        network_label = Label(network_controls_frame, padx=5, pady=5, bd=5, font="arial 16 bold", text="Edit Network:")
        network_label.pack()

        Button(network_controls_frame, padx=5, pady=5, text="Edit Layer", command=self.edit_network_layer).pack()

        b_add_layer = Button(network_controls_frame, padx=5, pady=5, text="Add Layer", command=self.add_network_layer)
        b_add_layer.pack()

        # *********** CONTROLS ***********
        controls_frame = Frame(self.window, padx=4, pady=4, bd=5, relief=RAISED)
        controls_frame.grid(row=1, column=0, columnspan=3, padx=frame_pad, pady=frame_pad)

        epoch_frame = Frame(controls_frame, bd=5)
        epoch_frame.grid(row=0, column=0, padx=10, pady=10)
        epoch_label = Label(epoch_frame, text="Number of Epochs:")
        epoch_label.grid(row=0, column=0, padx=10, pady=10, sticky=W)
        self.epoch_var = StringVar()
        self.epoch_var.set("5000")
        epoch_input = Entry(epoch_frame, textvariable=self.epoch_var)
        epoch_input.grid(row=1, column=0, sticky='s', padx=10, pady=10)

        summary_label = Label(epoch_frame, text="Summary Frequency:")
        summary_label.grid(row=0, column=1, padx=10, pady=10, sticky=W)
        self.summary_var = StringVar()
        self.summary_var.set("100")
        summary_input = Entry(epoch_frame, textvariable=self.summary_var)
        summary_input.grid(row=1, column=1, sticky='s', padx=10, pady=10)

        # b_make_net = Button(controls_frame, height=4, width=20,
        #                     command=lambda: self.make_network([2, 8,4,2, 2]), text="Make network")
        # b_make_net.grid(row=0, column=1, sticky='ns', padx=10, pady=10)
        b_train_net = Button(controls_frame, height=4, width=20, command=self.train_net, text="Train network")
        b_train_net.grid(row=0, column=2, sticky='ns', padx=10, pady=10)

        b_stop_net = Button(controls_frame, text="Stop", command=self.stop_network, height=4, width=10,)
        b_stop_net.grid(row=0, column=3, padx=10, pady=10)

        # *********** STATS ***********
        stats_frame = Frame(self.window, bd=5, relief=RAISED, padx=4,pady=4)
        stats_frame.grid(row=3, column=0, columnspan=3, padx=frame_pad, pady=frame_pad)

        self.progressbar = Progressbar(stats_frame, length=500)
        self.progressbar.grid(row=0, column=0, sticky='we', padx=10, pady=10)
        self.progress_label = Label(stats_frame, text="Press Train to begin")
        self.progress_label.grid(row=1, column=0, sticky='ns')

        self.make_network([2,3, 2])
        # start updates
        self.update_window()

    def update_window(self):
        update_time = 100

        if self.network and self.network.is_training:
            update_time = 10
            self.progressbar['value'] = 1 + (self.network.epoch / self.n_epochs) * 100
            if self.auto_update_weights and self.network.epoch % self.summary_freq == 0:
                self.update_tree_weights()
        self.progress_label['text'] = self.network.progress_string

        self.window.after(update_time, self.update_window)


        try:
            self.n_epochs = int(self.epoch_var.get())
        except ValueError:
            pass
        try:
            self.summary_freq = int(self.summary_var.get())
        except ValueError:
            pass

    def update_tree_weights(self):
        li = 1
        for layer in self.network.layers:
            layer_id = "Layer_" + str(li)
            ni = 0

            for neuron in layer.neurons:
                neuron_id = "Neuron_" + str(ni)
                # update delta:

                try:
                    item = layer_id + neuron_id

                    self.network_tree.item(item, text=neuron_id + " | Output: %.4f" % neuron.output)

                    item = layer_id + neuron_id + 'd'

                    self.network_tree.item(item, text="Delta: %.4f" % neuron.delta)
                except TclError:
                    pass

                # update weights                                                       TODO: output weights not updating
                wi = 0
                for weight in neuron.weights:

                    try:
                        item = layer_id + neuron_id + 'w' + str(wi)
                        self.network_tree.item(item, text="|%d|:\t %.4f" % (wi, weight))
                    except TclError:
                        pass
                    wi += 1
                ni += 1
            li += 1

    def make_network(self, architecture):
        input_size = architecture[0]
        output_size = architecture[-1]
        hidden_layers = architecture[1:-1]

        self.network = Network(input_size=input_size,
                               output_size=output_size,
                               learning_rate=learning_rate)

        for layer_size in hidden_layers:
            # hidden layers:
            self.network.add_layer(Dense(self.network, size=layer_size,
                                         activation_fn=sigmoid
                                         ))
        # output:
        self.network.add_layer(Dense(self.network, size=output_size, activation_fn=sigmoid))

        self.update_network_tree()
        self.update_window()

    def update_network_tree(self):
        self.network_tree.delete(*self.network_tree.get_children())

        # input:
        i_layer = self.network.input_layer
        layer_id = "Input_Layer"
        self.network_tree.insert('', 'end', iid="Input_Layer",
                                 text="Input_Layer" + " (" + i_layer.activation_fn.__name__ + ")")

        ni = -1
        for neuron in i_layer.neurons:
            ni += 1
            neuron_id = "Neuron_" + str(ni)
            self.network_tree.insert(layer_id, 'end',
                                     iid=layer_id + neuron_id,
                                     text=neuron_id + " | Output: %.4f" % neuron.output)


        li = 1
        for layer in self.network.layers:
            layer_id = "Layer_" + str(li)
            layer_label = "Output_Layer" if li == len(self.network.layers) else "Hidden_" + layer_id
            self.network_tree.insert('', 'end', iid=layer_id,
                                     text=layer_label + " ("+layer.activation_fn.__name__ + ")")
            ni=-1
            for neuron in layer.neurons:
                ni += 1
                neuron_id = "Neuron_" + str(ni)
                self.network_tree.insert(layer_id, 'end',
                                         iid=layer_id + neuron_id,
                                         text=neuron_id + " | Output: %.4f" % neuron.output)

                self.network_tree.insert(layer_id + neuron_id, 'end',
                                         iid=layer_id + neuron_id + 'w', text='Weights')

                self.network_tree.insert(layer_id + neuron_id, 'end',
                                         iid=layer_id + neuron_id + 'd', text='Delta: %.4f' % neuron.delta)
                wi = 0
                for weight in neuron.weights:
                    self.network_tree.insert(layer_id + neuron_id + 'w', 'end',
                                             iid=layer_id + neuron_id + 'w' + str(wi),
                                             text="|%d|:\t %.4f" % (wi, weight))
                    wi += 1
            li += 1

    def toggle_auto_update(self):
        self.auto_update_weights = not self.auto_update_weights
        self.t_auto_update['text'] = "Auto Update On" if self.auto_update_weights else "Auto Update Off"

    def train_net(self):

        self.update_window()
        inputs, labels, val_inputs, val_outputs = self.get_data()

        error_data, validation_data = self.network.train(inputs, labels,
                                                         self.n_epochs, self.summary_freq, DEBUG,
                                                         val_inputs, val_outputs)
        make_error_graph(error_data, validation_data)

    @staticmethod
    def get_data():
        data = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]])

        np.random.shuffle(data)

        inputs = data[:, 0:2]
        labels = data[:, -1]

        val_inputs = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        val_outputs = np.array([1, 0, 0, 1])

        return inputs, labels, val_inputs, val_outputs

    def stop_network(self):
        if self.network.is_training:
            self.network.stop = True

    def add_network_layer(self):

        pop = Toplevel(width=200, height=100)
        pop.geometry("300x200+750+100")

        Label(pop, text="New Layer:", font="arial 16 bold", padx=4, pady=4).grid(row=0, column=0, columnspan=2)
        Label(pop, text="Activation Function:", padx=4, pady=4).grid(row=1, column=0)

        function = StringVar()
        fn_choices = {'Sigmoid', 'ReLU', 'Dropout', 'Linear'}
        fn_dict = {'Sigmoid':sigmoid, 'ReLU': relu, 'Dropout':dropout, 'Linear': linear}

        function.set('Sigmoid')
        OptionMenu(pop, function, *fn_choices).grid(row=1, column=1, sticky=W)

        Label(pop, text="Nuerons:", padx=4, pady=4).grid(row=2, column=0, sticky=W)
        neurons_var = StringVar()
        neurons_var.set("3")
        Entry(pop, textvariable=neurons_var).grid(row=2, column=1, padx=4, pady=4, sticky=W)

        def add():
            self.network.add_layer(Layer(self.network), position=len(self.network.layers-2))
        Button(pop, text="Add", font="arial 12 bold", command=add)

    def edit_network_layer(self):
        selected_layer_name = self.network_tree.focus()
        print(selected_layer_name)
        selected_layer = None
        if selected_layer_name== "":
            return
        elif selected_layer_name == "Input_Layer":
            selected_layer = self.network.input_layer
        else:
            index = int(selected_layer_name[-1])
            selected_layer = self.network.layers[index-1]
        pop = Toplevel(width=200, height=100)
        pop.geometry("300x200+750+100")

        Label(pop, text="Edit " + selected_layer_name, font="arial 16 bold", padx=4, pady=4).grid(row=0, column=0, columnspan=2)
        Label(pop, text="Activation Function:", padx=4, pady=4).grid(row=1, column=0)

        function = StringVar()
        fn_choices = {'sigmoid', 'relu', 'dropout', 'linear'}
        fn_dict = {'sigmoid': sigmoid, 'relu': relu, 'dropout': dropout, 'linear': linear}

        function.set(selected_layer.activation_fn.__name__)
        OptionMenu(pop, function, *fn_choices).grid(row=1, column=1, sticky=W)

        Label(pop, text="Nuerons:", padx=4, pady=4).grid(row=2, column=0, sticky=W)
        neurons_var = IntVar()
        neurons_var.set(len(selected_layer.neurons))
        Entry(pop, textvariable=neurons_var).grid(row=2, column=1, padx=4, pady=4, sticky=W)

        def edit(gui):
            print(selected_layer.index)
            if selected_layer_name == "Input_Layer":
                gui.network.input_layer = Dense(self.network, size=int(neurons_var.get()),
                                                index=-1,
                                                activation_fn=fn_dict[function.get()]
                                                )
                # for i in range(gui.network.input_layer.size):
                #     gui.network.input_layer.neurons.append(Neuron(gui.network.input_layer, 0))
            else:
                print(fn_dict)
                new_layer = Layer(self.network, size=int(neurons_var.get()),
                                   index=selected_layer.index,
                                   activation_fn=fn_dict.get(function.get())
                                   )

                for i in range(int(neurons_var.get())):
                    n_weights = gui.network.input_size if selected_layer.index-1 ==0\
                        else len(gui.network.layers[selected_layer.index-1].neurons)
                    new_layer.neurons.append(
                        Neuron(layer=gui.network.layers[selected_layer.index-1],
                               n_weights=n_weights))
                    print(i)

                gui.network.layers[selected_layer.index-1] = new_layer

            gui.update_network_tree()

        Button(pop, text="Edit", font="arial 12 bold", command=lambda :edit(self)).grid(row=3, column=0, columnspan=2,
                                                                          padx=10, pady=10)
if __name__ == '__main__':
    GUI()
    # Let Tk take over
    mainloop()
