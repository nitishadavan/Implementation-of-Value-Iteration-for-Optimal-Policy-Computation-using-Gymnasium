# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
This project demonstrates the implementation of the Value Iteration algorithm in Reinforcement Learning using the Gymnasium library.

The objective is to find the optimal policy for an agent navigating through a 4×4 FrozenLake environment. The agent must move from the starting position to the goal while avoiding holes.

Value Iteration calculates the optimal state-value function and uses it to determine the best action for each state.





## Software Requirements
1.Python – Programming language

2.Gymnasium – Reinforcement Learning environment

3.NumPy – Numerical computations

4.Matplotlib – Data visualization

## Environment Description
The project uses the FrozenLake-v1 environment with a custom 4×4 grid.

The environment contains:

S – Starting position (State 3)

F – Frozen surface (safe position)

H – Hole (terminal state)

G – Goal position (State 12)

### Environment Layout

F  F  F  S

F  H  F  H

F  F  F  H

G  F  F  F

The agent starts at State 3 and must reach State 12 without falling into any holes.

The environment uses is_slippery=True, which introduces stochastic transitions. This means the agent may not always move in the intended direction.


## MDP Representation

The FrozenLake environment is represented as a Markov Decision Process (MDP), which consists of states, actions, transition probabilities, rewards, and a policy. The environment contains 16 states arranged in a 4×4 grid. The agent starts at State 3 and aims to reach the goal at State 12 while avoiding holes. It has four possible actions: Left, Down, Right, and Up. Since is_slippery=True, the transitions are stochastic, meaning the agent may not always move in the intended direction. The reward is given for reaching the goal, while other movements and falling into holes provide zero reward.

## Theory

Value Iteration is a dynamic programming algorithm used in Reinforcement Learning to find the optimal value function and policy for a Markov Decision Process. Initially, all state values are set to zero. The algorithm calculates the expected value of each possible action using the Bellman Optimality Equation and updates each state's value by selecting the maximum expected value. This process continues until the difference between successive updates becomes smaller than a predefined threshold. After convergence, the optimal policy is extracted by selecting the action with the highest expected value for each state.


## Algorithm

Value Iteration is a dynamic programming algorithm used to find the optimal value function and policy in a Markov Decision Process (MDP).

Steps Involved

Initialize the value function with zeros for all states.

Calculate the expected reward for each possible action.

Update the value of each state using the maximum expected action value.

Repeat the process until the value function converges.

Extract the optimal policy by selecting the action with the highest expected value for each state.



## Python Program

```python
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    P = env.unwrapped.P

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    iteration = 0

    while True:
        delta = 0

        for state in range(n_states):

            action_values = []

            for action in range(n_actions):

                value = 0

                for probability, next_state, reward, done in P[state][action]:

                    value += probability * (
                        reward + gamma * V[next_state] * (not done)
                    )

                action_values.append(value)

            best_value = max(action_values)

            delta = max(delta, abs(best_value - V[state]))

            V[state] = best_value

            # Update optimal policy
            policy[state] = np.argmax(action_values)

        iteration += 1

        if delta < theta:
            break

    return V, policy, iteration




```

---

## Output

<img width="1217" height="1292" alt="eb97ff9a-a93d-4594-9c59-78fe1d6b64fe" src="https://github.com/user-attachments/assets/a8d5e036-6e69-4524-92cc-889af67c24ea" />


## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The starting state was configured as State 3, and the goal state was configured as State 12. The algorithm calculated the state-value function and determined the optimal policy through repeated iterations until convergence. The final state values and optimal policy were displayed in a 4×4 grid format.

## Inference
The experiment demonstrates the successful implementation of the Value Iteration algorithm using the Gymnasium FrozenLake environment.



The algorithm calculates the optimal state values by repeatedly updating them based on expected future rewards.



The stochastic transitions make it challenging for the agent to reach the goal because it may move in an unintended direction.



The optimal policy helps the agent select the best possible action in each state.



The experiment shows that Value Iteration can determine an optimal policy by applying the Bellman Optimality Equation until the values converge.

