This is an implementation of cooperative baysian optimization in a 2D space

The intraction works as below:
    1. Agent chooses the arm x_t
    2. User observes the agent action and chooses the arm y_t
    3. The function is evaluated at point (x_t, y_t) and is shown to both
    4. The agent and the user update their belief accordingly

The goal is to optimize the function. The agent is an assistant helping the conservative user to find the optimum.