import math
import torch

from collections import Counter, defaultdict
from torch import nn

# uncomment if want to see each and every weight
#import numpy as np
#np.set_printoptions(threshold=np.inf)

GE='>='
LE='<='
COMPS = [GE, LE]

# Use the get_DNN class to grab the acasxu pytorch object. Must input a link to the nnet file, which
# located in ReluplexCav2017/nnet/ACASXU_run2a_"alpha"_"beta"_batch_2000.nnet
def load_rlv(rlv_infile):
    # This parser only makes really sense in the case where the network is a
    # feedforward network, organised in layers. It's most certainly wrong in
    # all the other situations.

    # What we will return:
    # -> The layers of a network in pytorch, corresponding to the network
    #    described in the .rlv
    # -> A corresponding list of the weights of the network
    # -> An input domain on which the property should be proved
    # -> A set of layers to stack on top of the network so as to transform
    #    the proof problem into a minimization problem.
    readline = lambda: rlv_infile.readline().strip().split(' ')

    all_layers = []
    layer_type = []
    nb_neuron_in_layer = Counter()
    neuron_depth = {}
    neuron_idx_in_layer = {}
    weight_from_neuron = defaultdict(dict)
    pool_parents = {}
    bias_on_neuron = {}
    network_depth = 0
    input_domain = []
    to_prove = []

    while True:
        line = readline()
        if line[0] == '':
            break
        if line[0] == "Input":
            n_name = line[1]
            n_depth = 0

            neuron_depth[n_name] = n_depth
            if n_depth >= len(all_layers):
                all_layers.append([])
                layer_type.append("Input")
            all_layers[n_depth].append(n_name)
            neuron_idx_in_layer[n_name] = nb_neuron_in_layer[n_depth]
            nb_neuron_in_layer[n_depth] += 1
            input_domain.append((-float('inf'), float('inf')))
        elif line[0] in ["Linear", "ReLU"]:
            n_name = line[1]
            n_bias = line[2]
            parents = [(line[i], line[i+1]) for i in range(3, len(line), 2)]

            deduced_depth = [neuron_depth[parent_name] + 1
                             for (_, parent_name) in parents]
            # Check that all the deduced depth are the same. This wouldn't be
            # the case for a ResNet type network but let's say we don't support
            # it for now :)
            for d in deduced_depth:
                assert d == deduced_depth[0], "Non Supported architecture"
            # If we are here, the deduced depth is probably correct
            n_depth = deduced_depth[0]

            neuron_depth[n_name] = n_depth
            if n_depth >= len(all_layers):
                # This is the first Neuron that we see of this layer
                all_layers.append([])
                layer_type.append(line[0])
                network_depth = n_depth
            else:
                # This is not the first neuron of this layer, let's make sure
                # the layer type is consistent
                assert line[0] == layer_type[n_depth]
            all_layers[n_depth].append(n_name)
            neuron_idx_in_layer[n_name] = nb_neuron_in_layer[n_depth]
            nb_neuron_in_layer[n_depth] += 1
            for weight_from_parent, parent_name in parents:
                weight_from_neuron[parent_name][n_name] = float(weight_from_parent)
            bias_on_neuron[n_name] = float(n_bias)
        elif line[0] == "Assert":
            # Ignore for now that there is some assert,
            # I'll figure out later how to deal with them
            ineq_symb = line[1]
            assert ineq_symb in COMPS
            off = float(line[2])
            parents = [(float(line[i]), line[i+1])
                       for i in range(3, len(line), 2)]

            if len(parents) == 1:
                # This is a constraint on a single variable, probably a simple bound.
                p_name = parents[0][1]
                depth = neuron_depth[p_name]
                pos_in_layer = neuron_idx_in_layer[p_name]
                weight = parents[0][0]
                # Normalise things a bit
                if weight < 0:
                    off = -off
                    weight = -weight
                    ineq_symb = LE if ineq_symb == GE else GE
                if weight != 1:
                    off = off / weight
                    weight = 1

                if depth == 0:
                    # This is a limiting bound on the input, let's update the
                    # domain
                    known_bounds = input_domain[pos_in_layer]
                    if ineq_symb == GE:
                        # The offset needs to be greater or equal than the
                        # value, this is an upper bound
                        new_bounds = (known_bounds[0], min(off, known_bounds[1]))
                    else:
                        # The offset needs to be less or equal than the value
                        # so this is a lower bound
                        new_bounds = (max(off, known_bounds[0]), known_bounds[1])
                    input_domain[pos_in_layer] = new_bounds
                elif depth == network_depth:
                    # If this is not on the input layer, this should be on the
                    # output layer. Imposing constraints on inner-hidden units
                    # is not supported for now.
                    to_prove.append(([(1.0, pos_in_layer)], off, ineq_symb))
                else:
                    raise Exception(f"Can't handle this line: {line}")
            else:
                parents_depth = [neuron_depth[parent_name] for _, parent_name in parents]
                assert all(network_depth == pdepth for pdepth in parents_depth), \
                "Only linear constraints on the output have been implemented."

                art_weights = [(weight, neuron_idx_in_layer[parent_name])
                               for (weight, parent_name) in parents]
                to_prove.append((art_weights, off, ineq_symb))
        else:
            print("Unknown start of line.")
            raise NotImplementedError

    # Check that we have a properly defined input domain
    for var_bounds in input_domain:
        assert not math.isinf(var_bounds[0]), "No lower bound for one of the variable"
        assert not math.isinf(var_bounds[1]), "No upper bound for one of the variable"
        assert var_bounds[1] >= var_bounds[0], "No feasible value for one variable"
    # TODO maybe: If we have a constraint that is an equality exactly, it might
    # be worth it to deal with this better than just representing it by two
    # inequality constraints. A solution might be to just modify the network so
    # that it takes one less input, and to fold the contribution into the bias.
    # Note that property 4 of Reluplex is such a property.

    # Construct the network layers
    net_layers = []
    nb_layers = len(all_layers) - 1
    for from_lay_idx in range(nb_layers):
        to_lay_idx = from_lay_idx + 1

        l_type = layer_type[to_lay_idx]
        nb_from = len(all_layers[from_lay_idx])
        nb_to = len(all_layers[to_lay_idx])

        if l_type in ["Linear", "ReLU"]:
            # If it's linear or ReLU, we're going to get a nn.Linear to
            # represent the Linear part, and eventually a nn.ReLU if necessary
            new_layer = torch.nn.Linear(nb_from, nb_to, bias=True)
            lin_weight = new_layer.weight.data
            # nb_to x nb_from
            bias = new_layer.bias.data
            # nb_to

            lin_weight.zero_()
            bias.zero_()

            for from_idx, from_name in enumerate(all_layers[from_lay_idx]):
                weight_from = weight_from_neuron[from_name]
                for to_name, weight_value in weight_from.items():
                    to_idx = neuron_idx_in_layer[to_name]
                    lin_weight[to_idx, from_idx] = weight_value
            for to_idx, to_name in enumerate(all_layers[to_lay_idx]):
                bias_value = bias_on_neuron[to_name]
                bias[to_idx] = bias_value

            net_layers.append(new_layer)
            if l_type == "ReLU":
                net_layers.append(torch.nn.ReLU())
        else:
            raise Exception("Not implemented")

    # The .rlv files contains the specifications that we need to satisfy for
    # obtaining a counterexample

    # We will add extra layers on top that will makes it so that finding the
    # minimum of the resulting network is equivalent to performing the proof.

    # The way we do it:
    # -> For each constraint, we transform it into a canonical representation
    #    `offset GreaterOrEqual linear_fun`
    # -> Create a new neuron with a value of `linear_fun - offset`
    # -> If this neuron is negative, this constraint is satisfied
    # -> We add a Max over all of these constraint outputs.
    #    If the output of the max is negative, that means that all of the
    #    constraints have been satisfied and therefore we have a counterexample

    # So, when we minimize this network,
    # * if we obtain a negative minimum,
    #     -> We have a counterexample
    # * if we obtain a positive minimum,
    #     -> There is no input which gives a negative value, and therefore no
    #        counterexamples


    # Make input_domain into a Tensor
    input_domain = torch.Tensor(input_domain)

    return net_layers, input_domain

with open("/home/socrates/proj_summer2018/output/output.rlv") as f:
    net_layers, domain = load_rlv(f)


total_weights = 0
#return net_layers, a nn.Linear object. Can find max weight. from doing something like:
lin_bias = net_layers[0].bias.data
lin_weights = net_layers[0].weight.detach().numpy()

print('*************************** input layer ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[1].bias.data
lin_weights = net_layers[1].weight.detach().numpy()

print('***************************  hidden layer 1 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[3].bias.data
lin_weights = net_layers[3].weight.detach().numpy()


print('***************************  hidden layer 2 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[5].bias.data
lin_weights = net_layers[5].weight.detach().numpy()


print('***************************  hidden layer 3 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[7].bias.data
lin_weights = net_layers[7].weight.detach().numpy()


print('***************************  hidden layer 3 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[9].bias.data
lin_weights = net_layers[9].weight.detach().numpy()

print('***************************  hidden layer 4 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[11].bias.detach().numpy()
lin_weights = net_layers[11].weight.detach().numpy()

print('***************************  hidden layer 5 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[13].bias.data
lin_weights = net_layers[13].weight.detach().numpy()
print('***************************  hidden layer 6 ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

lin_bias = net_layers[14].bias.data
lin_weights = net_layers[14].weight.detach().numpy()

print('*************************** output layer ***************************')
print('bias data: ',lin_bias)
print('weight data: ',lin_weights)
num_weight = len(lin_weights[0])*len(lin_weights)
print("number of weights in this layer = ", num_weight)
total_weights += num_weight

print("Overall structure (pytorch object)")
print(net_layers)
print("Input domain as given by rlv file")
print(domain)

print("Total number of edges with weights =",total_weights)
