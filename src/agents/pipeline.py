# Copyright 2018 Event Vision Library.

import time
import cv2
import torch

from bindsnet.encoding import bernoulli
from bindsnet.network.nodes import Input, AbstractInput
from bindsnet.network.monitors import Monitor

from env import events_to_image

class Pipeline(object):
    def __init__(self, network, environment, encoding, action_function, output):
        self.network = network
        self.env = environment
        if encoding == 'bernoulli':
            self.encoding = bernoulli
        self.action_function = action_function
        self.output = output
        
        # settings
        self.print_interval = 1
        self.save_interval = 1
        self.save_dir = 'network.pt'
        self.plot_length = 1.0
        self.plot_interval = 1
        self.history_length = 1

        self.time = 100
        self.dt = network.dt
        self.timestep = int(self.time / self.dt)

        # variables to use in this pipeline
        self.episode = 0
        self.iteration = 0
        self.accumulated_reward = 0
        self.reward_list = []

        self.obs = None
        self.reward = None
        self.done = None

        # add monitor into network
        for l in self.network.layers:
            self.network.add_monitor(Monitor(self.network.layers[l], 's', int(self.plot_length * self.plot_interval * self.timestep)),
                                        name='{:}_spikes'.format(l))
            if 'v' in self.network.layers[l].__dict__:
                self.network.add_monitor(Monitor(self.network.layers[l], 'v', int(self.plot_length * self.plot_interval * self.timestep)),
                                            name='{:}_voltages'.format(l))
        self.spike_record = {l: torch.Tensor().byte() for l in self.network.layers}
        self.set_spike_data()

        # Set up for multiple layers of input layers.
        self.encoded = {
            name: torch.Tensor() for name, layer in network.layers.items() if isinstance(layer, AbstractInput)
        }

        self.clock = time.time()

    def step(self):
        """
        Run an iteration of the pipeline.
        """
        if self.iteration % self.print_interval == 0:
            print('Iteration: {:} (Time: {:.4f})'.format(self.iteration, time.time() - self.clock))
            self.clock = time.time()

        #if self.iteration % self.save_interval == 0:
        #    print('Saving network to {:}'.format(self.save_dir))
        #    self.network.save(self.save_dir)

        # Choose action based on output neuron spiking.
        # need inserting to spike_record
        a = self.action_function(self, output=self.output)
        # convert number into action_name
        action_name = self.env.subject.action_list[a]

        # Run a step of the environment.
        events, self.reward, self.done, info = self.env.step(action=action_name)
        # currently image-based learning is adopted (Future work : spike-based)
        events_img = events_to_image(events, self.env.render_width, self.env.render_height)
        self.obs = torch.from_numpy(cv2.cvtColor(events_img, cv2.COLOR_BGR2GRAY)).float()/255.0

        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, time=self.time, dt=self.network.dt)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=self.encoded, time=self.time, reward=self.reward)
        self.set_spike_data() # insert into spike_record

        self.iteration += 1

        if self.done:
            self.iteration = 0
            self.episode += 1
            self.accumulated_reward = 0

    def set_spike_data(self):
        """
        Get the spike data from all layers in the pipeline's network.
        """
        self.spike_record = {l: self.network.monitors['{:}_spikes'.format(l)].get('s') for l in self.network.layers}


    def reset_(self):
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.iteration = 0
