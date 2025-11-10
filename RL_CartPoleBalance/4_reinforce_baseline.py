"""REINFORCE (with baseline) algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v0 import ActorModelV0
from src.models.critic import CriticModel

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.001

TotalTrainingReward = 0

if __name__ == "__main__":
    # Create environment, policy and critic
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel()
    actor_path = "./saved_models/actor_2.pt"
    critic_path = "./saved_models/critic_2.pt"

    # Training mode
    actor.train()
    critic.train()
    print(actor)
    print(critic)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Create optimizer with the critic parameters
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the REINFORCE with
    # baseline script

    count_conseq_successes = 0

    # Run infinitely many episodes
    training_iteration = 0
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()

        ## SAVE STATES FOR STATE VALUE TRAINING
        saved_states = list()

        # Prevent infinite loop
        reward_episode = 0
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)
            saved_states.append(state)

            reward_episode += reward

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------

        # Current discounted reward
        discounted_reward = 0.0

        # List of all the discounted rewards, for each time step
        discounted_rewards = list()

        # ---> TODO: compute discounted rewards
        val = 0
        for i in range(len(saved_probabilities) - 1, -1, -1):
            val = (
                saved_rewards[i] + val * DISCOUNT_FACTOR
            )  # Going from backwards. Well it is finite so it's okay to do it
            discounted_rewards.insert(0, val)

        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards, requires_grad=True)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        # COMPUTE STATE VALUES:
        state_values = list()
        for state in saved_states:
            state_values.append(critic(state))

        state_values = torch.tensor(state_values, requires_grad=True)

        # Update critic parameters
        # ------------------------------------------

        # Compute critic loss and update critic network
        critic_loss = F.mse_loss(state_values, discounted_rewards)

        # Reset gradients to 0.0
        critic_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        critic_loss.backward(retain_graph=True)

        # Update the critic parameters (gradient descent)
        critic_optimizer.step()

        # Update policy parameters
        # ------------------------------------------

        # For each time step
        actor_loss = list()
        for p, g, v in zip(saved_probabilities, discounted_rewards, state_values):

            # ---> TODO: compute policy loss
            time_step_actor_loss = 0.0
            time_step_actor_loss = (g - v) * torch.log(p) * -1
            # we do as the formula tells us. Blindly. Never doubt the math.

            actor_loss.append(time_step_actor_loss)

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        if episode_total_reward == 500:
            count_conseq_successes += 1
        else:
            count_conseq_successes = 0

        # Log results
        log_frequency = 5
        training_iteration += 1
        TotalTrainingReward += episode_total_reward
        if training_iteration % log_frequency == 0:

            # Save neural network
            torch.save(actor, actor_path)

            # Print results
            print(
                "iteration {} - last reward: {:.2f}".format(
                    training_iteration, episode_total_reward
                )
            )

            # ---> TODO: when do we stop the training?

        if count_conseq_successes > 15:
            print("Training complete! Enjoy the show!")
            print("Avg reward: ", TotalTrainingReward / training_iteration)
            break
