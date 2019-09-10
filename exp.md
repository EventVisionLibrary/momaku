
dqn_30fps: trackingだけ100fps. avoidanceは30fps


avoidanceは50fpsで追試


model2はbug。


# 3-7は50fps

```
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
import pickle

n_episode = 50

flist = ['reward3.pickle', 'reward4.pickle', 'reward5.pickle', 'reward6.pickle', 'reward7.pickle']

x = np.tile(np.arange(n_episode), (len(flist)))
y = np.zeros((n_episode * len(flist)))

for i, fname in enumerate(flist):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    y[n_episode * i: n_episode * i + n_episode] = data

df = pd.DataFrame(np.array([x, y]).T, columns=['episode', 'sum_reward'])
sns.lineplot(x='episode', y='sum_reward', data=df)
plt.savefig('avoidance_eval_exp.png')
```

# 8-13は100fps、target_update_interval=50


```
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
import pickle

n_episode = 50

flist = ['reward8.pickle', 'reward9.pickle', 'reward10.pickle', 'reward11.pickle', 'reward12.pickle', 'reward13.pickle']

x = np.tile(np.arange(n_episode), (len(flist)))
y = np.zeros((n_episode * len(flist)))

for i, fname in enumerate(flist):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    y[n_episode * i: n_episode * i + n_episode] = data

df = pd.DataFrame(np.array([x, y]).T, columns=['episode', 'sum_reward'])
sns.lineplot(x='episode', y='sum_reward', data=df)
plt.savefig('avoidance_eval_exp.png')
```

14は途中できっている


#15-はtarget_update_interval=200
15はミス。

#17-はtarget_update_interval=200


```
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
import pickle

n_episode = 50

flist = ['reward8.pickle', 'reward9.pickle', 'reward10.pickle', 'reward11.pickle', 'reward12.pickle', 'reward13.pickle']

x = np.tile(np.arange(n_episode), (len(flist)))
y = np.zeros((n_episode * len(flist)))

for i, fname in enumerate(flist):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    y[n_episode * i: n_episode * i + n_episode] = data

df = pd.DataFrame(np.array([x, y]).T, columns=['episode', 'sum_reward'])
sns.lineplot(x='episode', y='sum_reward', data=df)
plt.savefig('avoidance_eval_exp.png')
```

# Tracking


```
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
import pickle

n_episode = 50

flist = [
    'reward2.pickle',
    'reward3.pickle',
    'reward4.pickle',
    'reward5.pickle',
    'reward6.pickle',
    'reward10.pickle',
    'reward11.pickle']

x = np.tile(np.arange(n_episode), (len(flist)))
y = np.zeros((n_episode * len(flist)))

for i, fname in enumerate(flist):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    y[n_episode * i: n_episode * i + n_episode] = data

df = pd.DataFrame(np.array([x, y]).T, columns=['episode', 'sum_reward'])
sns.lineplot(x='episode', y='sum_reward', data=df)
plt.savefig('tracking_eval_exp.png')

```



