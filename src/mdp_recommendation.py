
import numpy as np


# ==========================================
# MOVIEMIND - MDP RECOMMENDATION
# ==========================================

# States represent user preference levels
states = ["Low", "Medium", "High"]

# Actions represent recommendation categories
actions = ["Comedy", "Action", "Drama"]

state_count = len(states)
action_count = len(actions)

# Transition probabilities
# P[current_state, action, next_state]
# These are illustrative assumptions for simulation.

transition = np.array([
    [
        [0.60, 0.30, 0.10],
        [0.40, 0.40, 0.20],
        [0.50, 0.30, 0.20]
    ],
    [
        [0.30, 0.50, 0.20],
        [0.20, 0.50, 0.30],
        [0.25, 0.50, 0.25]
    ],
    [
        [0.20, 0.30, 0.50],
        [0.10, 0.30, 0.60],
        [0.20, 0.30, 0.50]
    ]
])

# Illustrative rewards
# Higher reward means better user response.

rewards = np.array([
    [2, 3, 1],
    [2, 4, 3],
    [1, 5, 3]
])

# Discount factor
gamma = 0.90

# ==========================================
# VALUE ITERATION
# ==========================================

def value_iteration():

    values = np.zeros(state_count)

    iterations = 100

    for iteration in range(iterations):

        new_values = np.zeros(state_count)

        for state in range(state_count):

            action_values = []

            for action in range(action_count):

                expected_value = np.sum(
                    transition[state, action]
                    * values
                )

                q_value = (
                    rewards[state, action]
                    + gamma * expected_value
                )

                action_values.append(q_value)

            new_values[state] = max(action_values)

        values = new_values

    return values


# ==========================================
# EXTRACT OPTIMAL POLICY
# ==========================================

def get_optimal_policy(values):

    policy = {}

    for state in range(state_count):

        action_values = []

        for action in range(action_count):

            expected_value = np.sum(
                transition[state, action]
                * values
            )

            q_value = (
                rewards[state, action]
                + gamma * expected_value
            )

            action_values.append(q_value)

        best_action = np.argmax(action_values)

        policy[states[state]] = actions[best_action]

    return policy


# ==========================================
# TEST MDP
# ==========================================

if __name__ == "__main__":

    print("\n========== MOVIEMIND MDP ==========")

    optimal_values = value_iteration()

    optimal_policy = get_optimal_policy(
        optimal_values
    )

    print("\nOptimal State Values:")

    for state, value in zip(states, optimal_values):
        print(state, ":", round(value, 3))

    print("\nOptimal Recommendation Policy:")

    for state, action in optimal_policy.items():
        print(
            "User Preference:", state,
            "-> Recommend:", action
        )

    print("\nMDP RECOMMENDATION COMPLETED!")

