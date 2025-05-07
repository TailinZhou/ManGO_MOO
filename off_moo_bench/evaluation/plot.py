import os 
import matplotlib
import numpy as np 
import matplotlib.pyplot as plt 

def plot_y(y, save_dir, config,
           pareto_front=None, nadir_point=None, d_best=None, ideal_point=None, description=None):
    params = {
        'lines.linewidth': 1.5,
        'legend.fontsize': 22,
        'axes.labelsize': 32,
        'axes.titlesize': 36,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.scatterpoints': 1
    }
    matplotlib.rcParams.update(params)
    scatter_size = 200
    alpha = 0.02
    # if d_best is not None and d_best.shape[0] >= 6000:
    #     np.random.seed(0)
    #     _rand_indx = np.random.choice(d_best.shape[0], 6000, replace=False)
    #     d_best = d_best[_rand_indx]
    #     np.random.seed()

    plt.rc('font',family='DejaVu Sans')
    
    n_obj = len(y[0])
    
    if n_obj == 2:
        plt.figure(figsize=(10, 8))
        if d_best is not None:
            plt.scatter(d_best[:, 0], d_best[:, 1], alpha=alpha, color='grey', label='$\\mathcal{D}$(best)', s=scatter_size)
        plt.scatter(y[:, 0], y[:, 1], color='blue', label='Design Score', s=scatter_size)
        if pareto_front is not None:
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front', s=scatter_size, alpha=alpha)
        if nadir_point is not None:
            plt.scatter(nadir_point[0], nadir_point[1], color='green', label='Nadir Point', s=scatter_size)
        if ideal_point is not None:
            plt.scatter(ideal_point[0], ideal_point[1], color='black', label='Ideal Point', s=scatter_size)

        plt.xlabel(r'$y_1$', fontdict={'family' : 'DejaVu Sans'})
        plt.ylabel(r'$y_2$', fontdict={'family' : 'DejaVu Sans'})
        
    elif n_obj == 3:
        scatter_size = 50
        params = {
            'lines.linewidth': 1.5,
            'legend.fontsize': 12,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
        }
        matplotlib.rcParams.update(params)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if d_best is not None:
            ax.scatter(d_best[:, 0], d_best[:, 1], alpha=alpha, color='grey', label='$\\mathcal{D}$(best)', s=scatter_size)
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], color='blue', label='Design Score', s=scatter_size)
        if pareto_front is not None:
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color='red', label='Pareto Front', alpha=alpha, s=scatter_size)
        if nadir_point is not None:
            ax.scatter(nadir_point[0], nadir_point[1], nadir_point[2], color='green', label='Nadir Point', s=scatter_size)

        if ideal_point is not None:
            ax.scatter(ideal_point[0], ideal_point[1], ideal_point[2], color='black', label='Ideal Point', s=scatter_size)

        # ax.set_ylim([np.max(y[:, 1]), np.min(y[: 1])])
        
        ax.set_xlabel(r'$y_1$', fontdict={'family' : 'DejaVu Sans'}, labelpad=12)
        ax.set_ylabel(r'$y_2$', fontdict={'family' : 'DejaVu Sans'}, labelpad=12)
        ax.set_zlabel(r'$y_3$', fontdict={'family' : 'DejaVu Sans'})
        

    else:
        fig, axs = plt.subplots(n_obj, n_obj, figsize=(20, 20))
        for i in range(n_obj):
            for j in range(n_obj):
                if i == j:
                    continue
                ax = axs[i, j]
                ax.set_title(f'obj.{i + 1} and obj.{j + 1}',
                             fontdict={'family' : 'DejaVu Sans'})
                ax.set_xlabel(r'$y_{' + str(i) + '}$', fontdict={'family' : 'DejaVu Sans'})
                ax.set_ylabel(r'$y_{' + str(j) + '}$', fontdict={'family' : 'DejaVu Sans'})
                if d_best is not None:
                    ax.scatter(d_best[:, 0], d_best[:, 1], alpha=alpha, color='grey', label='$\\mathcal{D}$(best)', s=scatter_size)
                ax.scatter(y[:, i], y[:, j], color='blue', label='Design Score', s=scatter_size)
                if pareto_front is not None:
                    ax.scatter(pareto_front[:, i], pareto_front[:, j], color='red', label='Pareto Front', s=scatter_size)
                if nadir_point is not None:
                    ax.scatter(nadir_point[i], nadir_point[j], color='green', label='Nadir Point', s=scatter_size)

                if ideal_point is not None:
                    ax.scatter(ideal_point[i], ideal_point[j], color='black', label='Ideal Point', s=scatter_size)
 
                
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
    plt.title(f"Results of {config['task']}", fontdict={'family' : 'DejaVu Sans'})

    # 添加图例
    legend = plt.legend(ncol=2)

    # 修改图例中标记的透明度为1
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    if description is not None:
        plt.savefig(os.path.join(save_dir, f'{description}_pareto_front.png'))
        plt.savefig(os.path.join(save_dir, f'{description}_pareto_front.pdf'))
    else:
        plt.savefig(os.path.join(save_dir, 'pareto_front.png'))
        plt.savefig(os.path.join(save_dir, 'pareto_front.pdf'))
    plt.show()
    plt.close()