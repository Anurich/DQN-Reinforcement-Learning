# DQN-Reinforcement-Learning

<p> DQN is the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning.
 The model is a convolutional neural network, trained with a variant of Q-learning,
whose input is raw pixels and whose output is a value function estimating future
rewards. They apply DQN to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. DQN outperform all the previous approches.
</p>
<h2> Paper Linke </h2>
<a href="cs.toronto.edu/~vmnih/docs/dqn.pdf"> Playing Atari with Deep Reinforcement Learning </a>
<h2>Installation.</h2>
<ul> <li> <b>Python==3.6.6</b></li> <li>
<b>Pytorch==1.6.0</b></li></ul>
<h2> Game Environment.</h2>
<p> The game environment used is <b><a href="https://gym.openai.com/envs/MountainCar-v0/"> MountainCar-v0</a></b> with discreate action.</p>
<h2> Network Architecture</h2>
<p> Basic Network is used which consist of fully dense layer with activation function of relu and output layer with linear activation</p>
  <pre> <code> 
    def __init__(self, in_features, out_action):
        super(policyNetwork, self).__init__()
        self.dense =  nn.Linear(in_features = in_features, out_features = 512 )
        self.dense2 = nn.Linear(in_features= 512, out_features=64)
        self.output = nn.Linear(in_features =64, out_features=out_action)
    def forward(self, x):
        x = torch.relu(self.dense(x))
        x = torch.relu(self.dense2(x))
        output = self.output(x)
        return output

</code> </pre>
<h2> Result. </h2>
<figure>
<img src="gym.gif" alt="this slowpoke moves"  heigh="300" width="500"/>
</figure>
<h2> Important Note. </h2>
<p>
I am just the beginner and learning about this fascinating field, Please feel free to point out my mistake as well as feel free to contribute. 
Hope to upload more interesting project in future.
</p>

