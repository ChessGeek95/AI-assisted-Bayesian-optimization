import pickle
import numpy as np
from copy import deepcopy


class ExperimentSettings:
    """
    Responsible for generating experiment settings for different pairings of users and agents
    """
    def __init__(self, path=None):
        if path:
            self.load(path)
    
    def set_values(self, n_arms, n_trials, id):
        """
        n_arms: tuple(int, int)
            number of arms on the x and y axes, respectively.
        """
        self.n_arms = n_arms
        x, y = np.meshgrid(np.arange(self.n_arms[0]), np.arange(self.n_arms[1]))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.n_trials = n_trials
        self.id = id
        self.generate_trials()
        
    
    def generate_trials(self, func_type = 4):
        if func_type == 3:
            #self.function_args = np.random.random(size=(self.n_trials,4))*0.8 + 1.2
            #self.function_args *= [7, 3, 1.2, .5] #[9, 4, 1.7, 0.5]
            args = [[10,3,5,1.5], [4.5,10,2,3.5], [3,4,10,5], [2,3,5,10]]
            self.function_args = np.array([args[np.random.randint(4)] for i in range(self.n_trials)])
        elif func_type == 4:
            args = [[7, 3, 4], [3, 4, 7], [4, 7, 3]]
            self.function_args = np.array([args[np.random.randint(3)] for i in range(self.n_trials)])
        else:
            self.function_args = np.random.choice(4, size=(self.n_trials,))
        self.starting_points = np.random.randint(low=(0,0), high=self.n_arms, size=(self.n_trials,2))
        self.function_list = []
        self.user_data_list = []
        self.agent_data_list = []
        for i in range(self.n_trials):
            who = int(i*7/self.n_trials)
            function, user_data, agent_data = self.__generate_func(n_regions=2, 
                                                                    n_samples=5, 
                                                                    random_regions=False,
                                                                    function_type=func_type,
                                                                    args=self.function_args[i],
                                                                    who=who)
            self.function_list.append(function)
            self.user_data_list.append(user_data)
            self.agent_data_list.append(agent_data)

            """
            print("who:", who)
            print("agent_dt:", len(agent_data),agent_data)
            print("user_dt:", len(user_data),user_data)
            print("="*30)
            """
            
    
    
    def __generate_func(self, n_regions=2, n_samples=5, random_regions=False, function_type=3, args=None, who=2):
        """
        Generates a 2D function to optimize
        """
        if function_type == 0:
            return self._random_function(n_regions, n_samples, random_regions)
        elif function_type == 1:
            return self._periodic_function(n_regions, n_samples, random_regions, args)
        elif function_type == 2:
            return self._himmelblau_function(n_regions, n_samples, random_regions, args=args)
        elif function_type == 3:
            return self._custom_4modal_function(who, n_regions, n_samples, random_regions, args=args)
        elif function_type == 4:
            return self._custom_3modal_function(who, n_regions, n_samples, random_regions, args=args)
            
    
    def _random_function(self, n_regions=2, n_samples=2, random_regions=False):
        """
        Generates random 2D function
        """
        function_df = np.random.random(self.n_arms)

        if random_regions:
            user_data, agent_data = self._random_regions(n_samples)
        else:
            user_data, agent_data = self._split_regions(n_samples, n_regions)
        return function_df, user_data, agent_data


    def _periodic_function(self, n_regions=2, n_samples=5, random_regions=False, args=None):
        """
        Generates a generalized 2D priodic function
        """
        if args is None:
            args = 0.1 + 0.5 * np.random.random()
        n_arms_inv = (self.n_arms[1], self.n_arms[0])
        function_df = np.sin([args*(x+y) for x,y in self.all_points]).reshape(n_arms_inv)
        function_df += np.sin([args*(x*y) for x,y in self.all_points]).reshape(n_arms_inv)
        
        if random_regions:
            user_data, agent_data = self._random_regions(n_samples)
        else:
            user_data, agent_data = self._split_regions(n_samples, n_regions)
        return function_df, user_data, agent_data
        

    def _himmelblau_function(self, n_regions=2, n_samples=5, random_regions=False, args=None):
        """
        Implementation of generalized form of Himmelblau function
        Bring x and y to (-5,5) then
        f(x, y) = -(x^2 + y - 11)^2 - (x + y^2 - 7)^2 + ((x-x_i)**2 + (y-y_i)**2)
        assuming x_i and y_i are ith root of himmelblau function.
        Finally, we normalize and transfer!
        """
        p1 = (3, 2)
        p2 = (-2.805118, 3.131312)
        p3 = (-3.779310, -3.283186)
        p4 = (3.584428, -1.848126)
        roots = [p1, p2, p3, p4]
        #p = roots[np.random.randint(4)]
        #p = roots[1]
        self.glob_opt = args
        p = roots[args]
        
        f = lambda x,y: -1 * (x**2 + y - 11)**2 - 1 * (x + y**2 - 7)**2 - 10 * ((x-p[0])**2 + (y-p[1])**2)**0.5

        x, y = np.meshgrid(np.linspace(-5,5,self.n_arms[0]), np.linspace(-5,5,self.n_arms[1]))

        function_df = f(x,y)
        function_df -= np.min(function_df)
        function_df /= np.max(function_df)
        #function_df = 2 * function_df - 1
        function_df **= 3

        if random_regions:
            user_data, agent_data = self._random_regions(n_samples)
        else:
            user_data, agent_data = self._split_regions_to_four(n_samples, who=0)
        return function_df, user_data, agent_data

    
    
    def _custom_4modal_function(self, who, n_regions=2, n_samples=5, random_regions=False, args=None):
        p1 = (3, 2)
        p2 = (-2.805118, 3.131312)
        p3 = (-3.779310, -3.283186)
        p4 = (3.584428, -1.848126)
        roots = [p1, p2, p3, p4]
        m = args # shape of (4,)
        self.glob_opt = np.argmax(args)

        def filter(x,y,p):
            distance = ((x-p[0])**2 + (y-p[1])**2 + 0.5)**0.5
            flt = 1/(distance+1)
            return flt

        def f (x,y,p,m): 
            res_1 = -(x**2 + y - 11)**2 - (x + y**2 - 7)**2
            res_1 -= np.min(res_1)
            res_1 /= np.max(res_1)
            res_1 **= 6
            if m==0:
                res_2 = np.zeros_like(res_1)
            else:
                #res_2 = -(np.abs(x-p[0])**2 + np.abs(y-p[1])**2)**0.5
                res_2 = -(np.abs(x-p[0])**2 + np.abs(y-p[1])**2 + 1)**0.5
                res_2 -= np.min(res_2)
                res_2 /= np.max(res_2)
                res_2 **= 14
                res_2 = res_2  * m * filter(x,y,p)
            res = res_1 + res_2
            #res = res_2
            return res

        g = lambda x,y: f(x,y,roots[0], m[0]) + f(x,y,roots[1], m[1]) + f(x,y,roots[2], m[2]) + f(x,y,roots[3], m[3])

        x, y = np.meshgrid(np.linspace(-5.5,5.5,self.n_arms[0]), np.linspace(-5.5,5.,self.n_arms[1]))
        function_df = g(x,y)
        function_df -= np.min(function_df)
        function_df /= np.max(function_df)

        if random_regions:
            user_data, agent_data = self._random_regions(n_samples)
        else:
            user_data, agent_data = self._split_regions_to_four(deepcopy(function_df), n_samples, who)
        return function_df, user_data, agent_data

    
    def _custom_3modal_function(self, who, n_regions, n_samples=5, random_regions=False, args=None):
        p1 = (-0.35, 3.2)
        p2 = (-3, -0.75)
        p3 = (+2.7, -3.5)
        roots = [p1, p2, p3]

        m = args # shape of (3,)
        self.glob_opt = np.argmax(args)

        def filter(x,y,p):
            distance = ((x-p[0])**2 + (y-p[1])**2 + 0.5)**0.5
            flt = 1/(distance+1)
            return flt

        def f (x,y,p,m,w=0): 
            res_1 = -(x**2 + y - 11)**2 - (x + y**2 - 7)**2
            res_1 -= np.min(res_1)
            res_1 /= np.max(res_1)
            res_1 **= 6
            if m==0:
                res_2 = np.zeros_like(res_1)
            else:
                #res_2 = -(np.abs(x-p[0])**2 + np.abs(y-p[1])**2)**0.5
                res_2 = -(np.abs(x-p[0])**2 + np.abs(y-p[1])**2 + 1)**0.5
                res_2 -= np.min(res_2)
                res_2 /= np.max(res_2)
                res_2 **= 14
                res_2 = res_2  * m * filter(x,y,p)
            res = w*res_1 + res_2
            return res

        g = lambda x,y: f(x,y,roots[0], m[0]) + f(x,y,roots[1], m[1]) + f(x,y,roots[2], m[2]) + f(x,y,roots[2], 0, w=0.85)

        x, y = np.meshgrid(np.linspace(-5.5,5.5,self.n_arms[0]), np.linspace(-5.5,5.5,self.n_arms[1]))
        function_df = g(x,y)
        function_df -= np.min(function_df)
        function_df /= np.max(function_df)

        if random_regions:
            user_data, agent_data = self._random_regions(n_samples)
        else:
            user_data, agent_data = self._split_regions_to_roots(roots, n_samples, who)
        return function_df, user_data, agent_data

    

    def _random_regions(self, n_samples):
        """
        Generats random samples as the user and agent data
        """
        user_data = np.random.randint(low=(0, 0), high=self.n_arms, size=(n_samples,2))
        agent_data = np.random.randint(low=(0, 0), high=self.n_arms, size=(n_samples,2))
        return user_data, agent_data


    def _split_regions(self, n_samples, n_regions):
        """
        Splits the space into 4 and assign n_regions to each party.
        Then takes n_samples equally from those regions for each party.
        """
        mid = (int(self.n_arms[0]/2), int(self.n_arms[1]/2))
        regions = [[mid, self.n_arms],
                    [(0,mid[1]), (mid[0],self.n_arms[1])],
                    [(0,0), mid], 
                    [(mid[0],0), (self.n_arms[0], mid[1])]]
        agent_regions = (0, 1)
        user_regions = (1,2)
        agent_data = []
        for r in agent_regions:
            #samples = np.random.randint(low=regions[r][0], high=regions[r][1], size=(int(n_samples/n_regions),2))
            samples = np.random.multi(low=regions[r][0], high=regions[r][1], size=(int(n_samples/n_regions),2))
            for s in samples:
                agent_data.append(s)
    
        user_data = []
        for r in user_regions:
            samples = np.random.randint(low=regions[r][0], high=regions[r][1], size=(int(n_samples/n_regions),2))
            for s in samples:
                user_data.append(s)
        
        agent_data = np.array(agent_data)
        user_data = np.array(user_data)
        return user_data, agent_data



    def _split_regions_to_four(self, func_df, n_samples, who=2):
        """
        Splits the space into 4 and assign n_regions to each party.
        Then takes n_samples equally from those regions for each party.
        who == 0 : global opt assigned to agent
        who == 1 : global opt assigned to user
        who == 2 : global opt randomly assigned
        """
        mid = (int(self.n_arms[0]/2), int(self.n_arms[1]/2))
        n_arms = np.array(self.n_arms)-1
        regions = [ [mid, n_arms],
                    [(0,mid[1]), (mid[0],n_arms[1])],
                    [(0,0), mid], 
                    [(mid[0],0), (n_arms[0], mid[1])]]
        idxx = np.arange(4)
        idxx = idxx[idxx!=self.glob_opt]
        other_idx = np.random.choice(idxx)
        agent_regions = (self.glob_opt, other_idx)
        user_regions = idxx[idxx!=other_idx]

        # if who == 0: the AI knows the global, the user doesn't

        if who == 1: # the user knows the global, the AI doesn't
            agent_regions, user_regions = user_regions, agent_regions
        elif who == 2: # both of them know the global, the same regions
            user_regions = deepcopy(agent_regions)
        elif who == 3: # None of them know the global, the same regions
            agent_regions = deepcopy(user_regions)
        elif who == 4: # AI knows global and user has no initial knowledge
            user_regions = None
        elif who == 5: # AI knows local and user has no initial knowledge
            agent_regions = deepcopy(user_regions)
            user_regions = None
        elif who == 6: # None of them has initial knowledge
            agent_regions, user_regions = None, None
        
        def find_regmax(r_low, r_high):
            reg_func = np.zeros_like(func_df) - 100 #-np.inf
            reg_func[r_low[1]:r_high[1],r_low[0]:r_high[0]] = deepcopy(func_df[r_low[1]:r_high[1],r_low[0]:r_high[0]])
            idx_max = np.argmax(reg_func)
            argmax = np.array([idx_max%self.n_arms[0], idx_max//self.n_arms[0]])
            return argmax
        
        def draw_normal(r_low, r_high, n_samples, sig2):
            r_low, r_high = np.array(r_low), np.array(r_high)
            mu = find_regmax(r_low, r_high)
            if not isinstance(sig2, list):
                samples = np.random.multivariate_normal(mu, np.eye(2)*sig2, n_samples)
            else:
                samples = np.zeros((n_samples, 2))
                for i in range(n_samples):
                    samples[i] = np.random.multivariate_normal(mu+np.random.randint(int(5*i)+2, size=(2,)), np.eye(2)*sig2[i])
            samples = np.round(samples)
            samples = np.clip(samples, r_low, r_high).astype(int)
            return samples

        
        sig2 = [2, 6, 10]
        agent_data = []
        n_samples = [3,3]
        if who <= 5:
            for i,r in enumerate(agent_regions):
                if n_samples[i] > 0:
                    #samples = np.random.randint(low=regions[r][0], high=regions[r][1], size=(n_samples[i],2))
                    samples = draw_normal(regions[r][0], regions[r][1], n_samples[i], sig2)
                    for s in samples:
                        agent_data.append(s)
    
        user_data = []
        if who <= 3:
            for i,r in enumerate(user_regions):
                if n_samples[i] > 0:
                    #samples = np.random.randint(low=regions[r][0], high=regions[r][1], size=(n_samples[i],2))
                    samples = draw_normal(regions[r][0], regions[r][1], n_samples[i], sig2)
                    for s in samples:
                        user_data.append(s)
        user_data = np.array(user_data)
        agent_data = np.array(agent_data)
        return user_data, agent_data

    
    
    
    ###=====================================================================
    def _split_regions_to_roots(self, roots, n_samples, who=2):
        n_arms = np.array(self.n_arms)-1
        roots_arm = []
        for rt in roots:
            x_idx = np.round((rt[1] + 5.5)/11*(n_arms[0]-1))
            y_idx = np.round((rt[0] + 5.5)/11*(n_arms[1]-1))
            roots_arm.append((x_idx, y_idx))

        loc_idx = np.arange(3)
        loc_idx = loc_idx[loc_idx!=self.glob_opt]
        #print("glob:", self.glob_opt, "  loc:",loc_idx)

        # if who == 0: the AI knows the global, the user doesn't
        agent_region = self.glob_opt
        user_region = np.random.choice(loc_idx)

        if who == 1: # the user knows the global, the AI doesn't
            agent_region, user_region = user_region, agent_region
        elif who == 2: # both of them know the global, the same regions
            user_region = deepcopy(agent_region)
        elif who == 3: # None of them know the global, the same regions
            agent_region = deepcopy(user_region)
        elif who == 4: # AI knows global and user has no initial knowledge
            user_region = None
        elif who == 5: # AI knows local and user has no initial knowledge
            agent_region = deepcopy(user_region)
            user_region = None
        elif who == 6: # None of them has initial knowledge
            agent_region, user_region = None, None
        
        #print("AI reg:", agent_region, "  User reg:", user_region)

        def draw_normal(mu, sig2, n_samples):
            if not isinstance(sig2, list):
                samples = np.random.multivariate_normal(mu, np.eye(2)*sig2, n_samples)
            else:
                samples = []
                for i in range(n_samples):
                    samples.append(np.random.multivariate_normal(mu+np.random.randint(5*i+2, size=(2,))-2*i,
                                                                 np.eye(2)*sig2[i]))
            samples = np.round(samples)
            lb = np.clip(mu - n_arms//5, (0,0), n_arms).astype(int)
            ub = np.clip(mu + n_arms//5, (0,0), n_arms).astype(int)
            #print("bounds:", lb, ub, "   mu:", mu)
            samples = np.clip(samples, lb, ub).astype(int)
            return samples

        
        sig2 = [2, 4, 6, 8, 10]
        agent_data = []
        if agent_region is not None:
            mu = np.array(roots_arm[agent_region])
            samples = draw_normal(mu, sig2, n_samples)
            for s in samples:
                agent_data.append(s)
    
        user_data = []
        if user_region is not None:
            mu = np.array(roots_arm[user_region])
            samples = draw_normal(mu, sig2, n_samples)
            for s in samples:
                user_data.append(s)

        user_data = np.array(user_data)
        agent_data = np.array(agent_data)
        return user_data, agent_data
    ###=====================================================================
    
    
    
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.__dict__, file) 

    def load(self, path):
        with open(path, 'rb') as file:
            self.__dict__ = pickle.load(file)

    def display(self):
        pass