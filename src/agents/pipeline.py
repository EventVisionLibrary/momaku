import time

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

        # variables to use in this pipeline
        self.episode = 0
        self.iteration = 0
        self.accumulated_reward = 0
        self.reward_list = []

        self.obs = None
        self.reward = None
        self.done = None

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
        #a = self.action_function(self, output=self.output)
        
        # Run a step of the environment.
        self.obs, self.reward, self.done, info = self.env.step(action='forward')

        # Encode the observation using given encoding function.
        #for inpt in self.encoded:
        #    self.encoded[inpt] = self.encoding(self.obs, time=self.time, dt=self.network.dt, **kwargs)

        # Run the network on the spike train-encoded inputs.
        #self.network.run(inpts=self.encoded, time=self.time, reward=self.reward, **kwargs)

        self.iteration += 1

        if self.done:
            self.iteration = 0
            self.episode += 1
            self.accumulated_reward = 0


    def reset_(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.iteration = 0


    def reference(self):
        events, r, done, info = env.step(action='forward')
        image = events_to_image(events)
        print(image.shape)

        cv2.imwrite("../fig/image" + str(i) + ".png", image)
        print("Average Elapsed Time: {} s".format((time.time() - start) / N))