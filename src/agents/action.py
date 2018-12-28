import torch

def select_softmax(pipeline, output):
    """
    Selects an action using softmax function based on spiking from a network layer.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    # Sum of previous iterations' spikes (Not yet implemented)
    spikes = torch.sum(pipeline.spike_record[output], dim=1)
    _sum = torch.sum(torch.exp(spikes.float()))

    if _sum == 0:
        action = np.random.choice(pipeline.env.action_space.n)
    else:
        action = torch.multinomial((torch.exp(spikes.float()) / _sum).view(-1), 1)[0]

    return action