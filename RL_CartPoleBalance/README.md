# CartPoleBalance Reinforcement Learning Project

This project focuses on training a Reinforcement Learning agent to solve the CartPole balancing problem. The goal is to balance a pole on a cart for as long as possible.

## Project Structure

The project is organized as follows:

-   `src/`: Contains the source code for the project.
    -   `envs/`: Includes different versions of the CartPole environment.
    -   `models/`: Defines the actor and critic models used by the RL agents.
-   `documents/`: Contains relevant documents, such as the problem description and instructions.
-   `saved_models/`: Stores the trained models.
-   `*.py`: Python scripts for testing the environment, models, and training different RL agents.

## Getting Started

To get started with this project, you can run the training scripts to train the different RL agents and see how they perform on the CartPole task.

## RL Agents

This project implements and explores the following RL agents:

-   **REINFORCE**: A policy gradient method.
-   **REINFORCE with Baseline**: An improved version of REINFORCE that uses a baseline to reduce variance.
-   **Actor-Critic**: A method that uses two neural networks, an actor and a critic, to learn the policy and value function.

## Usage

You can use the following scripts to train and test the models:

-   `0_test_env.py`: Tests the CartPole environment.
-   `1_test_model.py`: Tests the initial model.
-   `2_reinforce.py`: Trains the REINFORCE agent.
-   `3_test_model.py`: Tests the trained REINFORCE model.
-   `4_reinforce_baseline.py`: Trains the REINFORCE with Baseline agent.
-   `5_test_model.py`: Tests the trained REINFORCE with Baseline model.
-   `6_actor_critic.py`: Trains the Actor-Critic agent.
-   `7_test_model.py`: Tests the trained Actor-Critic model.
