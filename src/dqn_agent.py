

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import pickle

class QFunction(chainer.Chain):
    def __init__(self, input_w, input_h, n_actions, hidden_dim=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=2, ksize=3)
            self.conv2 = L.Convolution2D(in_channels=2, out_channels=4, ksize=3)
            self.l1 = L.Linear(4 * (input_w - 4) * (input_h - 4), hidden_dim)
            self.l2 = L.Linear(hidden_dim, hidden_dim)
            self.l3 = L.Linear(hidden_dim, n_actions)
            self.bn1 = L.BatchNormalization(size=2)
            self.bn2 = L.BatchNormalization(size=4)
            self.bn3 = L.BatchNormalization(size=hidden_dim)
            self.bn4 = L.BatchNormalization(size=hidden_dim)

    def __call__(self, x):
        x = F.transpose(x, axes=(0, 3, 1, 2))
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.l1(h)))
        h = F.relu(self.bn4(self.l2(h)))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


def get_save_name(dir):
    i = 0
    agent_dir_name = os.path.join(dir, "model" + str(i) + ".pickle")
    while os.path.exists(agent_dir_name):
        agent_dir_name = os.path.join(dir, "model" + str(i) + ".pickle")
        i += 1
    reward_file_name = os.path.join(dir, "reward" + str(i - 1) + ".pickle")
    return (agent_dir_name, reward_file_name)

def save_all(dir, agent, R_list):
    agent_dir_name, reward_file_name = get_save_name(dir)
    print("Saving models to ", agent_dir_name)
    agent.save(agent_dir_name)
    with open(reward_file_name, "wb") as f:
        pickle.dump(R_list, f)

def draw_reward_fig(dir, R_list):
    plt.plot(np.array(R_list))
    plt.savefig(os.path.join(dir, "reward.png"))
    plt.close()
