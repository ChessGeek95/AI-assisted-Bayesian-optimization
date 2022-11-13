import enum
import numpy as np
import matplotlib

from baselines import UBC
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pathlib


PATH = str(pathlib.Path(__file__).parent.resolve())+'/'


def eval_ucb_scores(mu, std, beta):
    return mu + beta * std


def omnisci_baseline(function_list, user_data_list, agent_data_list
                        , init_gp_list, n_arms, max_iter=10):
    z_scores = []
    for trl in range(len(function_list)):
        ubc = UBC(function_list[trl], init_gp_list[trl], n_arms, 
                        user_data_list[trl], agent_data_list[trl])
        z_tmp = ubc.run(max_iter)
        z_scores.append(z_tmp)
    return np.array(z_scores)


def random_baseline(function_list, n_arms, max_iter=10):
    z_scores = []
    for trl in range(len(function_list)):
        function_df = function_list[trl]
        f_max = np.max(function_df)
        z_tmp = []
        for i in range(100):
            points = np.random.randint(low=(0,0), high=n_arms, size=(max_iter,2))
            z = function_df[points[:,1], points[:,0]]
            z_tmp.append(z)
        z_tmp = np.maximum.accumulate(z_tmp, axis=1)
        z_scores.append(np.mean(z_tmp/f_max * 100, axis=0))
    return np.array(z_scores)


def create_readme(path, *args):
    with open(path+"readme.md", "w") as f:
        for arg in args:
            arg_str = str(arg[0])+ ": " + str(arg[1]) + "\n"
            f.write(arg_str)




def plot_compare(scores_list, legends, name):
    x = np.arange(scores_list[0].shape[1])
    means = []
    sems = []
    for i, scores in enumerate(scores_list):
        mean = np.mean(scores, axis=0)
        sem = np.std(scores, axis=0)/(scores.shape[0]**0.5)
        sems.append(sem)
        means.append(mean)
        plt.plot(x, mean, ':', alpha=0.9)
        
    for i in range(len(legends)):
        plt.fill_between(x, means[i]-sems[i], means[i]+sems[i], alpha=0.2)

    plt.legend(legends)
    plt.savefig(PATH+'results/'+name+'.jpg')


def compare_trials(score_1, score_2, name):
    with open(PATH+"results/"+name+".txt", "w") as f:
        for i in range(score_2.shape[0]):
            f.write("="*30+"  "+str(i)+"  "+"="*30+"\n")
            f.write(str(np.round(score_1[i,:],1))+"\n")
            f.write(str(np.round(score_2[i,:],1))+"\n\n")



def compute_blur(mu): 
    return (mu - np.min(mu)+0.1)/(np.max(mu)- np.min(mu)+0.2)


def draw_arrow(A, B, color, alpha=0.5):
            plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
                    head_width=1.5, length_includes_head=True, linewidth=4, color=color, alpha=alpha)

def customize_axis(ax):
    axis_color = "#333333"
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.xaxis.label.set_color(axis_color)

def plot_vec(fig, gs, cur, itr, vec, pos, row, who, color):
    ax = fig.add_subplot(gs[row*14+2:row*14+6, pos])
    plt.plot(vec, '.', color=color, alpha=0.8)
    if itr > 0:
        plt.vlines(cur[who], np.min(vec), vec[cur[who]],
                linestyles ="solid", colors ="k",linewidth=4, alpha=0.3)
        plt.plot(cur[who], vec[cur[who]], 'X', color='k', markersize=8, alpha=0.5)
    #plt.ylim([-1.05,1.05])
    plt.ylabel(" ")
    customize_axis(ax)
    return ax

def plot_scatter(fig, gs, cur, trial, itr, x, y, mu, pos, who, color):
    x_arms, y_arms = trial.n_arms
    ax = fig.add_subplot(gs, aspect='equal')
    alphas = compute_blur(mu)
    plt.scatter(x, y, c=mu, cmap='Wistia', alpha=alphas, marker='.', s=alphas*25, rasterized=True)#, vmin=-1, vmax=1)

    if pos != 1:
        if who==0:
            plt.hlines(cur[1], 0, x_arms, linestyles="dashed", colors =color, linewidth=4, alpha=0.5)
            if itr==0:
                for point in trial.agent_data:
                    plt.plot(*point, '*', color=color, markersize=8)
        elif who==1:
            plt.vlines(cur[0], 0, y_arms, linestyles ="dashed", colors =color,linewidth=4, alpha=0.5)
            if itr==0:
                for point in trial.user_data:
                    plt.plot(*point, '*', color=color, markersize=8)
    
    else:    
        seq = [trial.init_point, *trial.xy_queries[:itr]]
        for i in range(len(seq[:-1])):
            p = seq[i]
            plt.plot(*p, 'X', color='k', markersize=8, alpha=0.6)
            draw_arrow(p, seq[i+1], color)
    plt.plot(*cur, 'X', color='k', markersize=10)
    if itr==0 and pos==1:
        plt.plot(*trial.argmax, 'o', color=color, markersize=15)

    customize_axis(ax)
    return ax


def subplot_border(fig, sub_gs, itr, border_color):
    ax = fig.add_subplot(sub_gs)
    ax.axis("off")
    p = ax.get_position().get_points().flatten()
    rect = patches.Rectangle((p[0]-0.14, p[2]-1.2),
                            (p[1]-p[0])+0.52, (p[3]-p[2])+1.8,
                            clip_on=False, linewidth=5, edgecolor=border_color, facecolor='none', alpha=0.5)
    ax.add_patch(rect)
    fig.text(p[0]-0.05, p[3]-0.17, "Agent's Turn => iter "+str(itr+1), fontsize=18, weight='bold', rotation='vertical', color=border_color)


def plot_all_trials(trials_planning, trials_greedy, name):
    pdf = PdfPages(PATH+'results/'+name+'.pdf')
    for i in range(len(trials_planning)):
        fig = plt.figure(figsize=(16,22))
        gs = fig.add_gridspec(35,3)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        trials_greedy[i].plot_scatter(0, fig, gs, row=0)
        trials_greedy[i].plot_scatter(-1, fig, gs, row=1)
        trials_planning[i].plot_scatter(-1, fig, gs, row=2)
        
        score_p = trials_planning[i].scores
        score_g = trials_greedy[i].scores
        ax = fig.add_subplot(gs[29:, 1])
        x = np.arange(1, score_p.shape[0]+1)
        ax.plot(x, score_p, 'o-', alpha=0.9)
        ax.plot(x, score_g, 'o-', alpha=0.9)
        ax.legend(["planning", "greedy"])

        plt.savefig(PATH+'results/'+name+'.jpg')
        pdf.savefig(fig, papertype = 'a4', orientation = 'portrait')
        plt.close()
    pdf.close()


"""
def plot_interaction(trial):#(user, agent, interface, who, row, fig, gs, iter):
    n_arms = (user.x_arms, user.y_arms)
    x = np.tile(np.arange(user.x_arms), user.y_arms)
    y = np.repeat(np.arange(user.y_arms), user.x_arms)
    mu_u, cov_u = user.current_prediction(cur=False)
    mu_a, cov_a = agent.current_prediction(cur=False)
    mu = interface.function
    cur = interface.cur
    
    user_color = "#cc0099"
    interface_color = "#248f24"
    agent_color = "#0066cc"

    if who==0:
        border_color = "#004d99"
        vec_u = mu_u.reshape(n_arms)[:,cur[1]].reshape((-1,))
        vec = mu.reshape(n_arms)[:,cur[1]].reshape((-1,))
        vec_a = mu_a.reshape(n_arms)[:,cur[1]].reshape((-1,))
    elif who==1:
        border_color = "#990073"
        vec_u = mu_u.reshape(n_arms)[cur[0]].reshape((-1,))
        vec = mu.reshape(n_arms)[cur[0]].reshape((-1,))
        vec_a = mu_a.reshape(n_arms)[cur[0]].reshape((-1,))

    comp_alpha = lambda mu: (mu - np.min(mu)+0.1)/(np.max(mu)- np.min(mu)+0.2)

    def draw_arrow(A, B, color):
        plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
                head_width=1.5, length_includes_head=True, linewidth=4, color=color, alpha=0.5)

    def customize_axis(ax):
        axis_color = "#333333"
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines['bottom'].set_color(axis_color)
        ax.spines['left'].set_color(axis_color)
        ax.tick_params(axis='x', colors=axis_color)
        ax.tick_params(axis='y', colors=axis_color)
        ax.yaxis.label.set_color(axis_color)
        ax.xaxis.label.set_color(axis_color)

    def plot_vec(vec, pos, color):
        ax = fig.add_subplot(gs[row*7:row*7+2, pos])
        plt.vlines(cur[who], np.min(vec), vec[cur[who]],
                    linestyles ="solid", colors ="k",linewidth=4, alpha=0.3)
        plt.plot(vec, '.', color=color, alpha=0.8)
        plt.plot(cur[who], vec[cur[who]], 'X', color='k', markersize=8, alpha=0.5)
        #plt.ylim([-1.05,1.05])
        plt.ylabel(" ")
        customize_axis(ax)

    def plot_scatter(mu, pos, color):
        ax = fig.add_subplot(gs[row*7+2:row*7+6, pos], aspect='equal')
        alphas = comp_alpha(mu)
        plt.scatter(x, y, c=mu, cmap='Wistia', alpha=alphas, marker='.', s=alphas*25)#, vmin=-1, vmax=1)

        if pos != 1:
            if who==0:
                plt.hlines(cur[1], 0, n_arms[0], linestyles="dashed", colors =color, linewidth=4, alpha=0.5)
            elif who==1:
                plt.vlines(cur[0], 0, n_arms[1], linestyles ="dashed", colors =color,linewidth=4, alpha=0.5)
        
        else:    
            seq = [interface.start, *interface.xy_queries]
            for i in range(len(seq[:-1])):
                p = seq[i]
                plt.plot(*p, 'X', color='k', markersize=8, alpha=0.6)
                if i % 2:
                    draw_arrow(p, seq[i+1], user_color)
                else:
                    draw_arrow(p, seq[i+1], agent_color)
        plt.plot(*cur, 'X', color='k', markersize=10)

        customize_axis(ax)

        return ax
    
    matplotlib.rcParams.update({'font.size': 16})

    plot_vec(vec_u, 0, user_color)
    if who==0:
        plt.title("User", fontsize=20, weight='bold')
    ax_1 = plot_scatter(mu_u, 0, user_color)
    
    
    plot_vec(vec, 1, interface_color)
    #plt.hlines(0, 0, n_arms[0], linestyles ="solid", colors ="#666666",linewidth=2, alpha=0.3)
    if who==0:
        plt.title("Interface", fontsize=20, weight='bold')
    ax_2 = plot_scatter(mu, 1, interface_color)
    plt.ylabel("User's action space", fontsize=22, weight='bold')
    plt.xlabel("Agent's action space", fontsize=22, weight='bold')
    
    plot_vec(vec_a, 2, agent_color)
    if row==0 and who==0:
        plt.title("Agent", fontsize=20, weight='bold')
    ax_3 = plot_scatter(mu_a, 2, agent_color)    
    
    if row==0 and who==0:
        p1 = ax_1.get_position().get_points().flatten()
        p3 = ax_3.get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[0], p1[3]+0.2, p3[2]-p1[0], 0.022])
        cbar = plt.colorbar(cax=ax_cbar, orientation='horizontal')
        cbar.set_ticks([])
        fig.text(p1[0]+0.005, p1[3]+0.205, 'Min', fontsize=18, weight='bold')
        fig.text(p3[2]-p1[0]+0.11, p1[3]+0.205, 'Max',color='white', fontsize=18, weight='bold')

    p1 = ax_1.get_position().get_points().flatten()
    rect = patches.Rectangle((-n_arms[0]*0.35, -n_arms[1]*0.23), n_arms[0]*5, n_arms[1]*1.95, 
                        clip_on=False, linewidth=5, edgecolor=border_color, facecolor='none', alpha=0.5)
    ax_1.add_patch(rect)

    if who==0:
        fig.text(p1[0]-0.11, p1[3]-0.16, "Agent's Turn => iter "+str(iter+1), fontsize=28, weight='bold', rotation='vertical', color=border_color)
    elif who==1:
        fig.text(p1[0]-0.11, p1[3]-0.16, "User's Turn => iter "+str(iter+1), fontsize=28, weight='bold', rotation='vertical', color=border_color)
"""