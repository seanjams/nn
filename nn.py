import json
import numpy as np
import random
from copy import deepcopy
from benchmark import st_time


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# def d_sigmoid(x):
#     return 1.0 / (2.0 + np.exp(-x) + np.exp(x))


def d_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def cost(output, desired_output):
    return np.sum(np.square(np.subtract(output, desired_output)))


def d_cost(output, desired_output):
    return 2 * (np.subtract(output, desired_output))


def visualize(input_data):
    """print image to command line"""
    for i in range(28):
        x = ""
        for j in range(28):
            index = 28 * i + j
            x += "1 " if input_data[index] else "0 "
        print(x)


def int_to_array(n):
    """ int_to_array(7) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] """
    return [1 if i == n else 0 for i in range(10)]


class Network:
    steps = []

    # training data
    training_input = []
    training_output = []

    # persisting data
    weights = []
    biases = []
    run_costs = []

    # data reset on every run
    activations = []
    zs = []
    d_sigmas = []
    d_activations = []
    d_weights = []
    d_biases = []

    @property
    def num_layers(self):
        return len(self.steps)

    def __init__(self, *args, **kwargs):
        """
        Create random set of matrices representing weights for the given steps.
        First element of these arrays is None for indexing purposes
        
        A0 -> F(W1, B1, A0) = Z1 -> Sigma(Z1) -> A1 ...
        """
        self.training_input = kwargs.get("training_input", [])
        self.training_output = kwargs.get("training_output", [])
        self.steps = kwargs.get("steps", [])
        self.init_weights()

    def init_weights(self):
        self.weights = [None] * self.num_layers
        self.biases = [None] * self.num_layers
        for i in range(1, self.num_layers):
            width = self.steps[i - 1]
            height = self.steps[i]
            weights = [[random.uniform(-1, 1) for _ in range(width)] for _ in range(height)]
            biases = [[random.uniform(-1, 1)] for _ in range(height)]
            self.weights[i] = np.array(weights)
            self.biases[i] = np.array(biases)

    def propagate(self, input_data):
        # init_activations
        self.activations = [None] * self.num_layers
        self.activations[0] = np.array([input_data]).T
        self.zs = [None] * self.num_layers
        self.d_sigmas = [None] * self.num_layers
        self.d_activations = [None] * self.num_layers
        self.d_weights = [None] * self.num_layers
        self.d_biases = [None] * self.num_layers

        for i in range(1, self.num_layers):
            activation = self.activations[i - 1]
            weights = self.weights[i]
            biases = self.biases[i]
            z = np.add(np.matmul(weights, activation), biases)
            # init data calculated now
            self.zs[i] = z
            self.activations[i] = sigmoid(z)
            self.d_sigmas[i] = d_sigmoid(z)

            # init data with zeros to be calculated in backprop
            self.d_weights[i] = np.zeros(self.weights[i].shape)
            self.d_biases[i] = np.zeros(self.biases[i].shape)
            self.d_activations[i] = np.zeros(z.shape)
    
    def back_propagate(self, output_data):
        output = self.activations[-1]
        desired_output = np.array([output_data]).T
        self.d_activations[-1] = d_cost(output, desired_output)
        for i in reversed(range(1, self.num_layers)):
            self.d_biases[i] = np.multiply(self.d_sigmas[i], self.d_activations[i])
            self.d_weights[i] = np.matmul(self.d_biases[i], self.activations[i - 1].T)

            if i > 1:
                for j in range(self.d_activations[i - 1].size):
                    weight_column = self.weights[i][:, j]
                    self.d_activations[i - 1][j] = np.dot(weight_column, self.d_biases[i])

    def network_cost(self, desired_output):
        if len(self.activations) < 2:
            return
        output = self.activations[-1]
        desired_output = np.array([desired_output]).T
        return np.sum(np.square(np.subtract(output, desired_output)))

    def run_stochastic(self, num_batches=1):
        input_data = self.training_input
        output_data = self.training_output
        batch_size = int(len(input_data) / num_batches)

        avg_batch_cost = 0
        avg_batch_weights = [None] * self.num_layers
        avg_batch_biases = [None] * self.num_layers

        for i in range(1, self.num_layers):
            avg_batch_weights[i] = np.zeros(self.weights[i].shape)
            avg_batch_biases[i] = np.zeros(self.biases[i].shape)

        for n in range(num_batches):
            start = n * batch_size
            end = (n + 1) * batch_size

            batch_cost = 0
            batch_weights = [None] * self.num_layers
            batch_biases = [None] * self.num_layers
            for i in range(1, self.num_layers):
                batch_weights[i] = np.zeros(self.weights[i].shape)
                batch_biases[i] = np.zeros(self.biases[i].shape)

            for i in range(start, end):
                # this loop should really go over the training data and run back_propagate for each
                # set of data, and then average the results of d_weights and d_biases before
                # applying to self.weights and self.biases.

                # feedforward
                self.propagate(input_data[i])
                
                # backprop
                self.back_propagate(output_data[i])

                # apply result to batch average of weights and biases
                batch_cost += self.network_cost(output_data[i])
                for i in range(1, self.num_layers):
                    batch_weights[i] = np.add(batch_weights[i], self.d_weights[i])
                    batch_biases[i] = np.add(batch_biases[i], self.d_biases[i])

            # save batch average
            avg_batch_cost += batch_cost / batch_size
            for i in range(1, self.num_layers):
                batch_weights[i] = batch_weights[i] / batch_size
                batch_biases[i] = batch_biases[i] / batch_size
                avg_batch_weights[i] = np.add(avg_batch_weights[i], batch_weights[i])
                avg_batch_biases[i] = np.add(avg_batch_biases[i], batch_biases[i])

        # learning step, apply batch averages to actual weights
        avg_batch_cost = avg_batch_cost / num_batches
        for i in range(1, self.num_layers):
            avg_batch_weights[i] = avg_batch_weights[i] / num_batches
            avg_batch_biases[i] = avg_batch_biases[i] / num_batches
            self.weights[i] = np.subtract(self.weights[i], avg_batch_weights[i])
            self.biases[i] = np.subtract(self.biases[i], avg_batch_biases[i])

        self.run_costs.append(avg_batch_cost)

    def run(self):
        input_data = self.training_input
        output_data = self.training_output
        if not len(input_data):
            return

        batch_cost = 0
        batch_weights = [None] * self.num_layers
        batch_biases = [None] * self.num_layers
        for i in range(1, self.num_layers):
            batch_weights[i] = np.zeros(self.weights[i].shape)
            batch_biases[i] = np.zeros(self.biases[i].shape)

        for i in range(len(input_data)):
            # this loop should really go over the training data and run back_propagate for each
            # set of data, and then average the results of d_weights and d_biases before
            # applying to self.weights and self.biases.

            # feedforward
            self.propagate(input_data[i])
            
            # backprop
            self.back_propagate(output_data[i])

            # apply result to batch average of weights and biases
            batch_cost += self.network_cost(output_data[i])
            for i in range(1, self.num_layers):
                batch_weights[i] = np.add(batch_weights[i], self.d_weights[i])
                batch_biases[i] = np.add(batch_biases[i], self.d_biases[i])

        # learning step, apply batch averages to actual weights
        batch_cost = batch_cost / len(input_data)
        for i in range(1, self.num_layers):
            batch_weights[i] = batch_weights[i] / len(input_data)
            batch_biases[i] = batch_biases[i] / len(input_data)
            self.weights[i] = np.subtract(self.weights[i], batch_weights[i])
            self.biases[i] = np.subtract(self.biases[i], batch_biases[i])

        self.run_costs.append(batch_cost)

    def test(self, input_data, output_data):
        for i in range(len(input_data)):
            self.propagate(input_data[i])
            output = self.activations[-1].T.tolist()[0]
            desired_output = output_data[i]
            answer = desired_output.index(1)
            success = False
            if output[answer] == max(output):
                success = True
            print(f"{answer}: {'Success' if success else 'Fail'}, {output}")


def test_single_example():
    # input data is random vector of non-zero decimal numbers.
    # output data is [1,0,0,0,...] of the same length.
    # This should train the network to make any entry gravitate towards the answer "0"
    steps = [2, 2, 2]
    input_data = [[random.uniform(0, 1) for _ in range(steps[0])]]
    output_data = [[1 if i == 0 else 0 for i in range(steps[-1])]]
    network = Network(steps=steps, training_input=input_data, training_output=output_data)
    # import ipdb; ipdb.set_trace()
    return network


def test_identity_example():
    # the five input vectors are the same as the desired output vectors.
    # this should train the network weights to converge towards the identity matrix.
    # NOTE In reality, the weights gravitate towards a matrix with negative entries except on the diagonal.
    # Extra credit: Why?
    steps = [5, 5]
    input_data = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    output_data = deepcopy(input_data)
    network = Network(steps=steps, training_input=input_data, training_output=output_data)
    # import ipdb; ipdb.set_trace()
    return network


def test_mnist_dataset():
    import gzip
    import pickle
    # import numpy as np
    # import random
    # from nn import Network
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train, val, test = u.load()
        input_data = train[0].tolist()
        output_data = [int_to_array(j) for j in train[1].tolist()]
        # test_input = [test[0][i,:].tolist() for i in range(10)]
        # test_output = [int_to_array(test[1][i]) for i in range(10)]

    steps = [len(input_data[0]), 16, 16, 10]
    network = Network(steps=steps, training_input=input_data, training_output=output_data)
    # import ipdb; ipdb.set_trace()
    return network

def save_pretrained_data(network):
    with open("pretrained_data.json", "w+") as f:
        result = {
            "steps": network.steps,
            "run_costs": network.run_costs
        }
        for i in range(1, network.num_layers):
            w = network.weights[i].tolist()
            b = network.biases[i].T.tolist()[0]
            result[f"w{i}"] = w
            result[f"b{i}"] = b
        f.writelines(json.dumps(result))

def load_pretrained_data(network=None):
    with open("pretrained_data.json") as f:
        pretrained_data = json.load(f)
        steps = pretrained_data.get("steps", [])
        run_costs = pretrained_data.get("run_costs", [])
        ignore_keys = ["steps", "run_costs"]
        weights = [(None, 0)]
        biases = [(None, 0)]
        for key in pretrained_data.keys():
            if key in ignore_keys:
                continue
            data = pretrained_data[key]
            index = int(key[1])
            if key.startswith("w"):
                weights.append((data, index))
            elif key.startswith("b"):
                biases.append((data, index))
        weights = [w[0] for w in sorted(weights, key=lambda i: i[1])]
        biases = [b[0] for b in sorted(biases, key=lambda i: i[1])]
        network = network or Network(steps=steps)
        network.weights = [np.array(w) for w in weights]
        network.biases = [np.array(b).T for b in biases]
        network.run_costs = run_costs
        return network

if __name__ == "__main__":
    test_mnist_dataset()