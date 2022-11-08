import numpy as np
import copy
from tqdm import tqdm
import cloudpickle
from agents.nodes import StateNode, ActionNode
from copy import deepcopy as cpy






class BAMCTS:
    """
    Base class for BAMCTS based on Bayes Adaptive Monte Carlo Tree Search for AI-assistant 
    in the defined 2D optimization task

    theta_u: tuple(np.array, np.array)
        posterior samples of the user model's parameters; alpha and beta, the unknown parts of MDP
    initial_obs: (dict)
        initial state of the tree, and the beliefs
    env: gym.env
        the optimization task environment
    K: float 
        exporation parameter of UCB
    """

    def __init__(self, theta_u, initial_obs, env, K):
        # Maybe it's better to initialize the node reward by the GP_AI
        self.alpha_u = theta_u[0]
        self.beta_u = theta_u[1]
        self.env = env
        self.K = K
        self.root = StateNode(state=initial_obs, is_root=True)
    

    def update_state_node(self, state_node, action_node):
        """
        Updates the decision node drawn by the select_outcome function.
        If it's a new node, it gets connected to the action node.
        Otherwise, returns the decsion node already stored in the action node's children.

        state_node: StateNode
            the state node to update
        action_node: ActionNode
            the action node of which the state node is a child
        """
        if state_node.state not in action_node.children.keys():
            state_node.parent = action_node
            action_node.add_children(state_node)
        else:
            state_node = action_node.children[state_node.state]

        return state_node


    def learn(self, n_sim, progress_bar=False):
        """
        Expand the tree and return the bet action

        n_sim: int
            number of tree traversals to do
        progress_bar: bool
            wether to show a progress bar (tqdm)
        """
        if progress_bar:
            iterations = tqdm(range(n_sim))
        else:
            iterations = range(n_sim)

        for _ in iterations:
            alpha_u = np.random.choice(self.alpha_u)
            beta_u = np.random.choice(self.beta_u)
            #alpha_u = 0.3
            #beta_u = 0.2
            self.grow_tree(alpha_u, beta_u)
            
    
    def grow_tree(self, alpha_u, beta_u):
        """
        Explores the current tree with the UCB principle until we reach an unvisited node
        where the reward is obtained with random rollouts.
        
        alpha_u: float
            user_model's search parameter
        beta_u: float
            user_model's updating parameter
        """
        state_node = self.root
        #self.env.update_usermodel_params(alpha_u, beta_u)
        internal_env = cpy(self.env) # could be reset instead
        internal_env.update_usermodel(alpha_u, beta_u)

        while (not state_node.is_final) and state_node.n_visits > 1:

            a = self.select(state_node)

            new_action_node = state_node.next_action_node(a)

            new_state_node, r = self.select_outcome(internal_env, new_action_node)

            new_state_node = self.update_state_node(new_state_node, new_action_node)

            new_state_node.reward = r
            new_action_node.reward = r

            state_node = new_state_node

        state_node.n_visits += 1
        cumulative_reward = self.evaluate(internal_env)

        """ 
        while not state_node.is_root:
            action_node = state_node.parent
            cumulative_reward += action_node.reward
            action_node.n_visits += 1    ########## I should still add lambda(y=0.2 for instance)
            action_node.cumulative_reward += (cumulative_reward - action_node.cumulative_reward)/action_node.n_visits
            state_node = action_node.parent
            state_node.n_visits += 1

        """
        while not state_node.is_root:
            action_node = state_node.parent
            cumulative_reward += action_node.reward
            action_node.cumulative_reward += cumulative_reward ##### cumulative reward should be mean, not sum
            action_node.n_visits += 1
            state_node = action_node.parent
            state_node.n_visits += 1
        
    

    def evaluate(self, env): # this function should be refined. it cannot go until the end in our case
        """
        Evaluates a state node playing until an terminal node using the rollot policy

        env: gym.env
            gym environemt that describes the state at the node to evaulate.
        :return: float
            the cumulative reward observed during the tree traversing.
        """
        max_iter = 100
        R = 0
        done = False
        iter = 0
        while ((not done) and (iter < max_iter)):
            iter += 1
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            R += r
        
        return R
    

    def select_outcome(self, env, action_node):
        """
        Given a ActionNode returns a StateNode

        env: gym.env
            the env that describes the state in which to select the outcome
        action_node: ActionNode 
            the action node from which selects the next state
        :return: StateNode
            the selected state node and corresponding reward
        """
        new_state_index, r, done, _ = env.step(action_node.action)        
        return StateNode(state=new_state_index, parent=action_node, is_final=done), r


    def select(self, state_node):
        """
        Selects the action to play from the current state node

        x: StateNode
            current state node
        :return: int
            action to play
        """
        if state_node.n_visits <= 2:
            state_node.children = {a: ActionNode(a, parent=state_node) for a in range(self.env.action_space.n)}

        def scoring(k):
            if state_node.children[k].n_visits > 0:
                return state_node.children[k].cumulative_reward/state_node.children[k].n_visits + \
                    self.K*np.sqrt(np.log(state_node.n_visits)/state_node.children[k].n_visits)
            else:
                return np.inf

        a = max(state_node.children, key=scoring)

        return a

    
    def best_action(self):
        """
        At the end of the simulations returns the most visited action

        :return: int 
            the best action according to the number of visits
        """
        number_of_visits_children = [node.n_visits for node in self.root.children.values()]
        index_best_action = np.argmax(number_of_visits_children)
        a = list(self.root.children.values())[index_best_action].action
        return a


    def forward(self, action, new_state):
        """
        If the env is determonostic we can salvage most of the tree structure.
        Advances the tree in the action taken if found in the tree nodes.

        action: int
        new_state: int
        """
        if action in self.root.children.keys():
            action_node = self.root.children[action]
            if len(action_node.children) > 1:
                self.root = StateNode(state=new_state, is_root=True)
            else:
                next_state_node = np.random.choice(list(action_node.children.values()))
                if next_state_node.state!=new_state:
                    raise RuntimeWarning("The env is probably stochastic")
                else:
                    next_state_node.parent = None
                    self.root.children.pop(action)
                    self.root = next_state_node
                    self.root.is_root = True
        else:
            raise RuntimeWarning("Action taken: {} is not in the children of the root node.".format(action))


    def _collect_data(self):
        """
        collects the data and parameters to save
        """
        data = {
            "K": self.K,
            "root": self.root
        }
        return data


    def save(self, path=None):
        """
        saves the tree structure as a pkl

        path: str 
            path where the tree is saved
        """
        data = self._collect_data()

        name = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f']+list(map(str, range(0, 10))), size=8)
        if path is None:
            path = './logs/'+"".join(name)+'_'
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        print("Saved at {}".format(path))


    def act(self):
        """
        returns the best action according to the maximum visits principle
        """
        action = self.best_action()
        return action








class BASPW(BAMCTS):
    """
    Simple Progressive Widening trees based on Monte Carlo Tree Search for Continuous and
    Stochastic Sequential Decision Making Problems, Courtoux

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha,
        where v is the number of visits to the current decision node
    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: exploration parameter of UCB
    """

    def __init__(self, alpha, theta_u, initial_obs, env, K):
        super().__init__(theta_u, initial_obs, env, K)
        self.alpha = alpha

    def select(self, state_node):
        """
        Selects the action to play from the current decision node. The number of children of a DecisionNode is
        kept finite at all times and monotonic to the number of visits of the DecisionNode.

        :param x: (DecisionNode) current decision node
        :return: (float) action to play
        """
        if state_node.n_visits**self.alpha >= len(state_node.children):
            a = self.env.action_space.sample()

        else:
            def scoring(k):
                if state_node.children[k].n_visits > 0:
                    return state_node.children[k].cumulative_reward/state_node.children[k].n_visits + \
                        self.K*np.sqrt(np.log(state_node.n_visits)/state_node.children[k].n_visits)
                else:
                    return np.inf

            a = max(state_node.children, key=scoring)

        return a

    def _collect_data(self):
        """
        Collects the data and parameters to save.
        """
        data = {
            "K": self.K,
            "root": self.root,
            "alpha": self.alpha
        }
        return data








class BADPW(BASPW):
    """
    Double Progressive Widening trees based on MCTS for Continuous and
        Stochastic Sequential Decision Making Problems, Courtoux.

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha,
        where v is the number of visits to the current decision node
    :param beta: (float) the number of outcomes of a random node is grater that v**beta,
        where v is the number of visits of the random node
    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: exploration parameter of UCB
    """
    def __init__(self, alpha, beta, theta_u, initial_obs, env, K):
        super().__init__(alpha, theta_u, initial_obs, env, K)
        self.beta = beta

    def select_outcome(self, env, action_node):
        """
        The number of outcomes of a RandomNode is kept fixed at all times and increasing
        in the number of visits of the random_node

        :param: random_node: (RandomNode) random node from which to select the next state
        :return: (DecisionNode, float) return the next decision node and reward
        """

        if action_node.n_visits**self.beta >= len(action_node.children):
            #if random_node.visits > len(random_node.children):
                #print('#',random_node.visits, len(random_node.children), random_node.visits**self.beta)
            new_state_index, r, done, _ = env.step(action_node.action)
            return StateNode(state=new_state_index, parent=action_node, is_final=done), r

        else:
            #if random_node.visits > len(random_node.children):
                #print('$',random_node.visits, len(random_node.children))
            unnorm_probs = [child.n_visits for child in action_node.children.values()]
            probs = np.array(unnorm_probs)/np.sum(unnorm_probs)

            chosen_state = np.random.choice(list(action_node.children.values()), p=probs)
            return (chosen_state, chosen_state.reward)
