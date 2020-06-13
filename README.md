# qrep - Q Learning and Representation Analysis

In this project I train a deep Q learner to play pong, and achieve a +19.4
score after 2 million frames of training. Although this is a near perfect
score, a shallow analysis reveals that the learner takes advantage of the
deterministic nature of the pong AI, and exploits a weakness to quickly score
point after point in rote fashion.

Please follow requirements.txt to install the required packages to train. I
trained on an NVIDIA GeForce RTX 2060 Super, and I averaged about 3 hours/1
million frames. You will need to run the following commands to get started:

```
# clone this repo and cd into the main folder
$ mkdir checkpoints; mkdir logs 
$ pip install requirements 
$ python run_dqn_pong.py --num_frames 2000000 --learning_rate 0.00002 
        --gamma 0.995 --replay_buffer 200000 --epsilon_decay 100000 
        --copy_frequency 40000
```

Thank you to Ge Shi for providing starter code. Pretty much everything in
Wrappers is his code, and much of dqn.py, though I claim authorship for 
run_dqn_pong.py
