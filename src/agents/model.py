# Copyright 2018 Event Vision Library.

def create_network(network='bindsnet', input_width=80, input_height=80, action_num=3):
    if network == 'bindsnet':
        return create_bindsnet(input_width, input_height, action_num)

def create_bindsnet(input_width, input_height, action_num=3):
    from bindsnet.network import Network
    from bindsnet.learning import MSTDP
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    network = Network(dt=1.0)
    # Layers of neurons.
    inpt = Input(n=input_height * input_width, shape=[input_height, input_width], traces=True)
    middle = LIFNodes(n=100, traces=True)
    out = LIFNodes(n=action_num, refrac=0, traces=True)

    # Connections between layers.
    inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
    middle_out = Connection(source=middle, target=out, wmin=0, wmax=1, update_rule=MSTDP, nu=1e-1, norm=0.5 * middle.n)

    # Add all layers and connections to the network.
    network.add_layer(inpt, name='Input Layer')
    network.add_layer(middle, name='Hidden Layer')
    network.add_layer(out, name='Output Layer')
    network.add_connection(inpt_middle, source='Input Layer', target='Hidden Layer')
    network.add_connection(middle_out, source='Hidden Layer', target='Output Layer')
    return network
