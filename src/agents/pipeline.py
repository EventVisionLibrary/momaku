import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from bindsnet.network.nodes import Input, AbstractInput
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

from env import events_to_image

class Pipeline(object):
    def __init__(self, network, environment, encoding, action_function, output):
        self.network = network
        self.env = environment
        self.encoding = encoding
        self.action_function = action_function
        self.output = output
        
        # settings
        self.print_interval = 1
        self.save_interval = 1
        self.save_dir = 'network.pt'
        self.plot_length = 1.0
        self.plot_interval = 1
        self.history_length = 1

        # time
        self.time = 100
        self.dt = network.dt
        self.timestep = int(self.time / self.dt)

        # variables to use in this pipeline
        self.episode = 0
        self.iteration = 0
        self.accumulated_reward = 0
        self.reward_list = []

        # for plot
        self.plot_type = "color"
        self.s_ims, self.s_axes = None, None
        self.v_ims, self.v_axes = None, None
        self.obs_im, self.obs_ax = None, None
        self.reward_im, self.reward_ax = None, None
        
        self.obs = None
        self.reward = None
        self.done = None
        self.action_name = None

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
        self.action_name = self.env.subject.action_list[a]

        # Run a step of the environment.
        events, self.reward, self.done, info = self.env.step(action=self.action_name)

        # reward accumulation
        self.accumulated_reward += self.reward

        # currently image-based learning is adopted (Future work : spike-based)
        events_img = events_to_image(events, self.env.render_width, self.env.render_height)
        self.obs = torch.from_numpy(cv2.cvtColor(events_img, cv2.COLOR_BGR2GRAY)).float()/255.0

        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, time=self.time, dt=self.network.dt)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=self.encoded, time=self.time, reward=self.reward)
        self.set_spike_data() # insert into spike_record

        # Plot relevant data.
        if self.iteration % self.plot_interval == 0:
            self.plot_data()
            self.plot_obs()

        self.iteration += 1

        if self.done:
            self.iteration = 0
            self.episode += 1
            self.reward_list.append(self.accumulated_reward)
            self.accumulated_reward = 0
            self.plot_reward()

    def plot_obs(self):
        """
        Plot the processed observation after difference against history
        """
        if self.obs_im is None and self.obs_ax is None:
            fig, self.obs_ax = plt.subplots()
            self.obs_ax.set_title('Observation')
            self.obs_ax.set_xticks(())
            self.obs_ax.set_yticks(())
            self.obs_im = self.obs_ax.imshow(self.obs, cmap='gray')
        else:
            self.obs_im.set_data(self.obs)

    def plot_reward(self):
        """
        Plot the change of accumulated reward for each episodes
        """
        if self.reward_im is None and self.reward_ax is None:
            fig, self.reward_ax = plt.subplots()
            self.reward_ax.set_title('Reward')
            self.reward_plot, = self.reward_ax.plot(self.reward_list)
        else:
            reward_array = np.array(self.reward_list)
            y_min = reward_array.min()
            y_max = reward_array.max()
            self.reward_ax.set_xlim(left=0, right=self.episode)
            self.reward_ax.set_ylim(bottom=y_min, top=y_max)
            self.reward_plot.set_data(range(self.episode), self.reward_list)


    def plot_data(self):
        """
        Plot desired variables.
        """
        # Set latest data
        self.set_spike_data()
        self.set_voltage_data()

        # Initialize plots
        if self.s_ims is None and self.s_axes is None and self.v_ims is None and self.v_axes is None:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record,
                    plot_type=self.plot_type, threshold=self.threshold_value)
        else:
            # Update the plots dynamically
            self.s_ims, self.s_axes = plot_spikes(self.spike_record, ims=self.s_ims, axes=self.s_axes)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record, ims=self.v_ims,
                    axes=self.v_axes, plot_type=self.plot_type, threshold=self.threshold_value)

        plt.pause(1e-8)
        plt.show()

    def set_spike_data(self):
        """
        Get the spike data from all layers in the pipeline's network.
        """
        self.spike_record = {l: self.network.monitors['{:}_spikes'.format(l)].get('s') for l in self.network.layers}

    def set_voltage_data(self):
        """
        Get the voltage data and threshold value from all applicable layers in the pipeline's network.
        """
        self.voltage_record = {}
        self.threshold_value = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                self.voltage_record[l] = self.network.monitors['{:}_voltages'.format(l)].get('v')
            if 'thresh' in self.network.layers[l].__dict__:
                self.threshold_value[l] = self.network.layers[l].thresh

    def reset_(self):
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.iteration = 0