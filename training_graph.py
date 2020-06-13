#python run_dqn_pong.py --lp models/pretrained1.pth --lr .00002 --g .995 --rb 200000 --ed 100000 --cf 40000
import pickle
import numpy as np
import matplotlib.pyplot as plt

run0 = "1000k_1e-05_0.99_100k_50k_30"
run1 = "900k_2e-05_0.995_200k_40k_100"

logs = []
for i, log_name in enumerate([run0, run1]):
    with open("./logs/" + log_name + ".pkl", "rb") as f:
        log = pickle.load(f)
        for frame_idx, logged_info in log.items():
            logs.append((frame_idx + i*1000000,) + tuple([logged_info[frame_idx] for frame_idx in logged_info]))

logs = np.array(logs)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(logs[:,0], logs[:,3], s=10, c='b', marker='s', label='Last-10 Avg Rewards')
ax1.set_ylabel('Last-10 Avg Rewards', c='b')
ax1.set_xlabel('Frame Index')

ax2 = ax1.twinx()
ax2.scatter(logs[:,0], logs[:,1], s=10, c='r', marker='o', label='Avg Loss')
ax2.set_ylabel('Cumulative Average Loss', c='r')
ax2.set_title('Reward and Loss During DQN Training')
fig.savefig('reward_and_loss_graph.jpg', format='jpeg', dpi=400, bbox_inches='tight')

