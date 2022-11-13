from sqlite3 import IntegrityError
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
from utils import PATH




class Trial:
    def __init__(self, path=None):
        if path:
            self.load(path)

    def set_values(self, user, agent, interface, experiment_settings, n_iters, episode):
        self.n_iters = n_iters
        self.n_arms = interface.n_arms
        self.init_point = experiment_settings.starting_points[episode]
        self.function_df = experiment_settings.function_list[episode]
        self.user_data = experiment_settings.user_data_list[episode]
        self.agent_data = experiment_settings.agent_data_list[episode]
        self.init_gp = agent.init_gp
        self.user_params = (user.alpha, user.beta)
        idx_max = np.argmax(self.function_df)
        self.argmax = (idx_max%self.n_arms[0], idx_max//self.n_arms[0])
        self.user_beliefs = []
        self.agent_beliefs = []
        self.scores = None
        self.xy_queries = None
        self.z_queries = None

    def add_prior_data(self, agent_data, user_data):
        self.agent_data_z = agent_data[1]
        self.user_data_z = user_data[1]    

    def add_belief(self, user_belief, agent_belief, user_model_beleif=None):
        user_belief = (user_belief[0].astype(np.float16), user_belief[1].astype(np.float16))
        agent_belief = (agent_belief[0].astype(np.float16), agent_belief[1].astype(np.float16))
        self.user_beliefs.append(user_belief)
        self.agent_beliefs.append(agent_belief)
    
    def add_queries(self, xy_queries, z_queries, scores):
        self.xy_queries = xy_queries
        self.z_queries = z_queries
        self.scores = scores

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.__dict__, file) 

    def load(self, path):
        with open(path, 'rb') as file:
            self.__dict__ = pickle.load(file)

    def plot_compare_to(self, other, name):
        pdf = PdfPages(PATH+'results/'+name+'.pdf')
        for itr in range(self.n_iters):
            fig = plt.figure(figsize=(22,16))
            gs = fig.add_gridspec(19,3)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)
            self.plot_scatter(itr, fig, gs, row=0)
            other.plot_scatter(itr, fig, gs, row=1)
            pdf.savefig(fig, papertype = 'a4', orientation = 'landscape')
            plt.close()
        pdf.close()

    def plot(self, path=None):
        pdf = PdfPages(PATH+'results/multipage_pdf.pdf')
        for itr in range(2):#self.n_iters):
            fig = plt.figure(figsize=(22,16))
            gs = fig.add_gridspec(20,3)
            #fig.tight_layout()
            plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)
            self.__plot_interaction(itr, fig, gs)
            pdf.savefig(fig, papertype = 'a4', orientation = 'landscape')
            plt.close()
        pdf.close()

    def __plot_interaction(self, itr, fig, gs):
        x_arms, y_arms = self.n_arms
        x = np.tile(np.arange(x_arms), y_arms)
        y = np.repeat(np.arange(y_arms), x_arms)
        mu_u = self.user_beliefs[itr][0]
        mu_a = self.agent_beliefs[itr][0]
        mu = self.function_df
        cur = self.init_point if itr==0 else self.xy_queries[itr-1]
        #print(mu.shape)
        
        vec_u_1 = mu_u.reshape((y_arms, x_arms))[cur[1],:].reshape((-1,))
        vec_1 = mu.reshape((y_arms, x_arms))[cur[1],:].reshape((-1,))
        vec_a_1 = mu_a.reshape((y_arms, x_arms))[cur[1],:].reshape((-1,))
        #print(vec_u_1.shape)
    
        vec_u_2 = mu_u.reshape((y_arms, x_arms))[:,cur[0]].reshape((-1,))
        vec_2 = mu.reshape((y_arms, x_arms))[:,cur[0]].reshape((-1,))
        vec_a_2 = mu_a.reshape((y_arms, x_arms))[:,cur[0]].reshape((-1,))
        #print(vec_u_2.shape)
        
        #======== colors ===========
        user_color = "#cc0099"
        interface_color = "#248f24"
        agent_color = "#0066cc"
        border_color_1 = "#004d99"
        border_color_2 = "#990073"
        arrow_color = "#248f24"
        
        #===========================
                

        matplotlib.rcParams.update({'font.size': 16})
        ax_1 = utils.plot_vec(fig, gs[7:15, 0], cur, itr, vec_u_1, 0, 0, 1, user_color)
        plt.title("User", fontsize=18, weight='bold')
        utils.plot_scatter(fig, gs, cur, self, itr, x, y, mu_u, 0, 1, user_color)
        utils.plot_vec(fig, gs, cur, itr, vec_u_2, 0, 1, 0, user_color)
        
        utils.plot_vec(fig, gs, cur, itr, vec_1, 1, 0, 1, interface_color)
        #plt.hlines(0, 0, n_arms[0], linestyles ="solid", colors ="#666666",linewidth=2, alpha=0.3)
        plt.title("Interface", fontsize=18, weight='bold')
        utils.plot_scatter(fig, gs[7:15, 1], cur, self, itr, x, y, mu, 1, -1, interface_color)
        plt.ylabel("User's action space", fontsize=20, weight='bold')
        plt.xlabel("Agent's action space", fontsize=20, weight='bold')
        utils.plot_vec(fig, gs, cur, itr, vec_2, 1, 1, 0, interface_color)
        
        ax_2 = utils.plot_vec(fig, gs[7:15, 2], cur, itr, vec_a_1, 2, 0, 1, agent_color)
        plt.title("Agent", fontsize=18, weight='bold')
        utils.plot_scatter(fig, gs, cur, self, itr, x, y, mu_a, 2, 0, agent_color)
        utils.plot_vec(fig, gs, cur, itr, vec_a_2, 2, 1, 0, agent_color)
        
        p1 = ax_1.get_position().get_points().flatten()
        #print(p1)
        p3 = ax_2.get_position().get_points().flatten()
        #print(p3)
        ax_cbar = fig.add_axes([p1[0], p1[3]+0.1, p3[2]-p1[0], 0.022])
        #print(ax_cbar)
        cbar = plt.colorbar(cax=ax_cbar, orientation='horizontal')
        cbar.set_ticks([])
        fig.text(p1[0]+0.005, p1[3]+0.107, 'Min', fontsize=18, weight='bold')
        fig.text(p3[2]-p1[0]+0.067, p1[3]+0.107, 'Max',color='white', fontsize=18, weight='bold')


        utils.subplot_border(fig, gs[2:6, :], itr, border_color_1)
        
        ax = fig.add_subplot(gs[16:20, :])
        ax.axis("off")
        p = ax.get_position().get_points().flatten()
        rect = patches.Rectangle((p[0]-0.14, p[2]-1.2),
                                (p[1]-p[0])+1.08, (p[3]-p[2])+2.06,
                                clip_on=False, linewidth=5, edgecolor=border_color_2, facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        
        fig.text(p[0]-0.05, p[3]-0.18, "User's Turn => iter "+str(itr+1), fontsize=18, weight='bold', rotation='vertical', color=border_color_2)

    
    def plot_scatter(self, itr, fig, gs, row):
        if itr == -1:
            itr = self.n_iters-1
        x_arms, y_arms = self.n_arms
        x = np.tile(np.arange(x_arms), y_arms)
        y = np.repeat(np.arange(y_arms), x_arms)
        mu_u = self.user_beliefs[itr][0]
        mu_a = self.agent_beliefs[itr][0]
        mu = self.function_df
        #print("mu_u",itr, mu_u.shape)
        #print("mu_a",itr, mu_a.shape)
        #print("mu",itr, mu.shape)
        cur = self.init_point if itr==0 else self.xy_queries[itr-1]
        #print(mu.shape)
        
        #======== colors ===========
        user_color = "#cc0099"
        interface_color = "#248f24"
        agent_color = "#0066cc"
        
        #===========================
        
        matplotlib.rcParams.update({'font.size': 16})
        ax_1 = utils.plot_scatter(fig, gs[row*9+2:row*9+10, 0], cur, self, itr, x, y, mu_u, 0, 1, user_color)
        plt.title("User", fontsize=18, weight='bold')
        
        utils.plot_scatter(fig, gs[row*9+2:row*9+10, 1], cur, self, itr, x, y, mu, 1, -1, interface_color)
        plt.title("Interface", fontsize=18, weight='bold')
        plt.ylabel("User's action space", fontsize=20, weight='bold')
        plt.xlabel("Agent's action space", fontsize=20, weight='bold')
        
        ax_2 = utils.plot_scatter(fig, gs[row*9+2:row*9+10, 2], cur, self, itr, x, y, mu_a, 2, 0, agent_color)
        plt.title("Agent", fontsize=18, weight='bold')

        #utils.subplot_border(fig, gs[row*9+2:row*9+10, :], itr, interface_color)
        
        if row == 0:
            p1 = ax_1.get_position().get_points().flatten()
            p3 = ax_2.get_position().get_points().flatten()
            ax_cbar = fig.add_axes([p1[0], p1[3]+0.1, p3[2]-p1[0], 0.022])
            cbar = plt.colorbar(cax=ax_cbar, orientation='horizontal')
            cbar.set_ticks([])
            fig.text(p1[0]+0.005, p1[3]+0.107, 'Min', fontsize=18, weight='bold')
            fig.text(p3[2]-p1[0]+0.067, p1[3]+0.107, 'Max',color='white', fontsize=18, weight='bold')
