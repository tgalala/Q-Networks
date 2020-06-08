# Reinforcement Learning
# Q-Learning Applications: Basic and Advanced Applications Using OpenAI Environments
<br>

<b>Language</b> <br>
Python

<b>OpenAI Gym environment</b><br><br>
<b>Taxi-v3</b> <br>
This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.
[Dietterich2000]	T Erez, Y Tassa, E Todorov, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition", 2011. <br>
<br><b>Algorithm</b><br>
Q-learning
<br><center>
<img src="https://github.com/tgalala/Reinforcement-Learning-Q-Learning-Applications/blob/master/images/taxi.png?raw=true" height="200">
</center>
<b>LunarLander-v2</b> <br>
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine. <br>
<b>Algorithm</b>
Deep Q-learning  |  Double Deep Q-learning  |   Dueling Double Deep Q-learning <br>
<center>
<img src="https://github.com/tgalala/Reinforcement-Learning-Q-Learning-Applications/blob/master/images/lunar.png?raw=true" height="200">
</center>

<br>
<b>Introduction</b> <br>
This work presents the methods and tools used to build and apply Q-learning and DQN algorithms (and their variations)  in two separate OpenAI Gym environments (Taxi v_3, Lunar_Lander v_2) to demonstrate their effectiveness as model free methods for training goal driven agents. The tasks are navigation based and episodic, but include both discrete  and continuous environments. We seek to identify and investigate the most important methods and parameters for improving performance. We achieved our best results with Dueling Double DQN by solving the lunar landing environment in 298 episodes.

<b>Index Terms</b> <br>
Q-learning, Deep Q-Learning, Double Deep Q-Learning, Dueling Double Deep Q-Learning, Taxi V3, Lunar Landing V2


<br><br>

<img src="https://github.com/tgalala/Reinforcement-Learning-Q-Learning-Applications/blob/master/images/algorithms1.jpg?raw=true" width="800">
<Br>
<img src="https://github.com/tgalala/Reinforcement-Learning-Q-Learning-Applications/blob/master/images/algorithms2.jpg?raw=true" width="810"> 


