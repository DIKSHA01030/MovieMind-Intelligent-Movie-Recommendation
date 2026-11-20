
import numpy as np
from sklearn.tree import DecisionTreeRegressor


# ==========================================
# MOVIEMIND - FITTED VALUE ITERATION
# ==========================================

states = ["Low", "Medium", "High"]
actions = ["Comedy", "Action", "Drama"]

state_count = len(states)
action_count = len(actions)

# Illustrative transition probabilities
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

rewards = np.array([
    [2, 3, 1],
    [2, 4, 3],
    [1, 5, 3]
])

gamma = 0.90
iterations = 20

# State features for the regression model
X = np.arange(state_count).reshape(-1, 1).astype(float)

# Initial value estimates
values = np.zeros(state_count)

# ==========================================
# FITTED VALUE ITERATION
# ==========================================

model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

for iteration in range(iterations):

    targets = []

    for state in range(state_count):

        action_values = []

        for action in range(action_count):

            expected_future_value = np.sum(
                transition[state, action] * values
            )

            q_value = (
                rewards[state, action]
                + gamma * expected_future_value
            )

            action_values.append(q_value)

        targets.append(max(action_values))

    # Fit regression model to Bellman targets
    model.fit(X, np.array(targets))

    # Approximate values using the fitted model
    values = model.predict(X)

# ==========================================
# FIND OPTIMAL POLICY
# ==========================================

policy = {}

for state in range(state_count):

    action_values = []

    for action in range(action_count):

        expected_future_value = np.sum(
            transition[state, action] * values
        )

        q_value = (
            rewards[state, action]
            + gamma * expected_future_value
        )

        action_values.append(q_value)

    best_action = np.argmax(action_values)

    policy[states[state]] = actions[best_action]


# ==========================================
# DISPLAY RESULTS
# ==========================================

if __name__ == "__main__":

    print("\n========== FITTED VALUE ITERATION ==========")

    print("\nEstimated State Values:")

    for state, value in zip(states, values):
        print(state, ":", round(value, 3))

    print("\nOptimal Recommendation Policy:")

    for state, action in policy.items():
        print(
            "User Preference:", state,
            "-> Recommend:", action
        )

    print("\nIterations:", iterations)

    print("\nFITTED VALUE ITERATION COMPLETED!")
