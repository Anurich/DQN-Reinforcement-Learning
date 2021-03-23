from dqn import *
import numpy as np
from PIL import Image
def test():
    # load the model
    state_size  = env.reset().shape[0]
    action_size = env.action_space.n
    model = main(state=state_size, action= action_size)
    checkpoint = torch.load("model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    frames = []
    for i in range(10):
        initial_state = env.reset()
        done = False
        total_reward = 0.0
        while True:
            frames.append(Image.fromarray(env.render(mode="rgb_array")))
            action = model.take_action(initial_state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            initial_state = next_state
            if done:
                break
        print(total_reward)
    env.close()
    frames[0].save(fp="gym.gif", format='GIF', append_images=frames[1:],
         save_all=True, duration=20, loop=0)
test()
