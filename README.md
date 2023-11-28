[//]: # (Image References)

## Introduction

This project marks the inaugural step in Udacity's Deep Reinforcement Learning Nanodegree. The primary objective involves training an agent to navigate and gather bananas within a vast square world.

The reward system is straightforward: +1 for collecting a yellow banana and -1 for collecting a blue one. The agent's mission is to accumulate as many yellow bananas as possible while steering clear of the blue ones.

The state space comprises 37 dimensions, including the agent's velocity and a ray-based perception of objects in the agent's forward direction. With this information, the agent learns to make optimal decisions among four discrete actions:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Solving the environment is episodic, requiring the agent to achieve an average score of +13 over 100 consecutive episodes.

### Method

The key steps outlined in Navigation.ipynb include:
1. Agent initialization. 
2. Evaluation of state and action space.
3. Learning through Deep Q-Networks (DQN) with a simple 3-layered neural network model. 
4. Iterative training until the agent reaches a threshold score of 15.0.

## Future work ideas

The roadmap for future enhancements encompasses the following strategies:

### Double Deep Q-Networks (DDQN)

Addressing the issue of overestimated Q-values in Deep Q-Networks, this approach involves using two sets of parameters (w and w'). One set selects the best action, while the other evaluates that action, mitigating the risk of propagating inaccurately high rewards.

### Prioritized Experience Replay

To enhance learning efficiency, prioritized Experience Replay involves replaying important transitions more frequently. This departs from the traditional uniform sampling, allowing the agent to focus on more significant experiences.

### Dueling Agents

Dueling networks adopt a dual-stream structure to estimate both the state value function V(s) and the advantage for each action A(s,a). These values are then combined to derive the desired Q-values.

## Getting Started

1. Download the environment compatible with your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Unzip the downloaded file into the working folder.

3. Commence work by opening and running Navigation.ipynb.