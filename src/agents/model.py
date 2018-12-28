from bindsnet.network import Network
from bindsnet.learning import MSTDP
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

IMG_WIDTH=80
IMG_HEIGHT=80

def create_network(action_num=3):
    network = Network(dt=1.0)

    # Layers of neurons.
    inpt = Input(n=IMG_HEIGHT*IMG_WIDTH, shape=[IMG_HEIGHT, IMG_WIDTH], traces=True)
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