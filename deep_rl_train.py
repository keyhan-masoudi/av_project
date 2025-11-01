from controllers.simulator import Simulator
from controllers.loader import Loader
from controllers.zone_managers.deepRL.deep_rl_agent import DeepRLAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from utils.clock import Clock
from models.node.cloud import CloudNode
from config import Config


def train_rl_agent(episodes=500, batch_size=32, update_target_every=10):
    """
    Trains the Deep RL agent for task offloading.
    """
    # Initialize environment and agent
    loader = Loader(
        zone_file="./data/hamburg.zon.xml",
        fixed_fn_file="./data/hamburg.fn.xml",
        mobile_file="./data/vehicles",
        task_file="./data/tasks",
        checkpoint_path="./checkpoints",
    )

    cloud = CloudNode(
        id="CLOUD0",
        x=Config.CloudConfig.DEFAULT_X,
        y=Config.CloudConfig.DEFAULT_Y,
        power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        remaining_power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        radius=Config.CloudConfig.DEFAULT_RADIUS,
    )

    simulator = Simulator(loader, Clock(), cloud)
    # print(f"test:{simulator}")
    env = DeepRLEnvironment(simulator)
    agent = DeepRLAgent(state_dim=6, action_dim=3)

    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Save trained model
    agent.save_model()
    print("Training complete. Model saved as deep_rl_model.pth")


if __name__ == "__main__":
    train_rl_agent()
