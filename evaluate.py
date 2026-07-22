import torch
import time
from rocket_env import RocketEnv
from agent import RocketNetwork

# 1. Initialize environment with human render mode
env = RocketEnv(render_mode="human")
state_dim, action_dim = 6, 6

# 2. Load trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RocketNetwork(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("rocket_dqn.pth", map_location=device))
model.eval()

# 3. Run evaluation episodes
num_eval_episodes = 5

for episode in range(num_eval_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    print(f"\n--- Evaluation Episode {episode + 1} ---")

    while not done:
        # Render the current frame
        env.render()

        # Choose greedy action (no random exploration)
        state_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()

        # Step physics
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        obs = next_obs
        total_reward += reward

        # Small delay so the animation runs at a visible speed
        time.sleep(0.05)

    print(f"Episode {episode + 1} finished with Total Reward: {total_reward:.2f}")

env.close()