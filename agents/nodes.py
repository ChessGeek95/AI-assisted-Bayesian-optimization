
class StateNode:
    """
    The State Node class in the BA-MC tree

    :param state: (int) defining the state
    :param parent: (ActionNode) The ancestor action node in the tree
    :param is_root: (bool)
    :param is_final: (bool)
    """


    def __init__(self, state=None, parent=None, is_root=False, is_final=False):
        self.n_visits = 0
        self.reward = 0
        self.state = state
        self.parent = parent
        self.is_final = is_final
        self.is_root = is_root
        self.children = {}


    def add_children(self, action_node):
        """
        Adds a ActionNode object to the dictionary of children (key is the action)

        :param random_node: (ActionNode) add an action node to the set of children
        """

        self.children[action_node.action] = action_node


    def next_action_node(self, action):
        """
        Returns successor action node. If not exists, first adds as a child and then returns.

        :param action: (int) the actiuon taken at the current node
        :return: (ActionNode) the successor action node
        """

        if action not in self.children.keys():
            new_action_node = ActionNode(action, parent=self)
            self.add_children(new_action_node)
        else:
            new_action_node = self.children[action]
        return new_action_node


    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if k == "children":
                pass
            elif k == "parent":
                pass
            else:
                s += str(k)+": "+str(v)+"\n"
        return s




class ActionNode:
    """
    The ActionNode class defined by the state and the action, representing an intermediate state after taking an action and before transitioning to the next state

    :param action: (action) the action taken in the parent state node
    :param parent: (StateNode)
    """


    def __init__(self, action, parent=None):
        self.n_visits = 0
        self.cumulative_reward = 0
        self.action = action
        self.parent = parent
        self.children = {}


    def add_children(self, state_node):
        """
        Adds an ActionNode object to the dictionary of children (key is the state)

        :param state_node: (StateNode) the state node to add to the children dict
        """

        self.children[state_node.state] = state_node


    def __repr__(self):
        mean_rew = round(self.cumulative_reward/(self.n_visits+1), 2)
        s = "action: {}\nmean_reward: {}\nvisits: {}".format(self.action, mean_rew, self.n_visits)
        return s
