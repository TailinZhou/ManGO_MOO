import numpy as np
import torch
from scipy.spatial.distance import cdist
import os,sys
BASE_PATH = os.path.join(
   '/home/tzhouaq/offline-moo/'
)
sys.path.append(BASE_PATH)
import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from utils import set_seed, get_quantile_solutions
from off_moo_bench.evaluation.metrics import hv, igd
from off_moo_bench.evaluation.plot import plot_y


def get_device(device: str = "auto") -> torch.device:
    """
    Returns specified device.
    Input:
        device: device. Default auto.
    Returns:
        The specified device.
    """
    if device.lower() != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def save_indicators(HV_list, IGD_list, y_cand, y_gen, y_given_list, pf, description, logging_dir, kkk, seed, config, task):
    nadir_point = task.nadir_point #手动在benchmark中已经设置好了
    ideal_point = task.problem.ideal_point
    if config['normalize_ys']:
        ideal_point = task.normalize_y(ideal_point)
        nadir_point = task.normalize_y(nadir_point)

    _, d_best = task.get_N_non_dominated_solutions(
        N=task.x.shape[0],  # int(len_data * config["data_preserved_ratio"]),#config["num_solutions"],
        return_x=True, return_y=True
    )
    d_best_hv = hv(nadir_point, d_best, config['task'])
    ideal_hv= hv(nadir_point, ideal_point, config['task'])
    y_given_hv = hv(nadir_point, y_given_list, config['task'])

    # nan_indices = np.where(np.isnan(y_cand).all(axis=1))[0]
    # y_cand = y_cand[~np.isnan(y_cand).any(axis=1)]


    hv_value = hv(nadir_point, y_cand, config['task'])
    pf_hv = hv(nadir_point, pf, config['task'])


    print(f"Hypervolume (100th): {hv_value:4f}")
    HV_list.append(hv_value)
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    print(f"Ideal Hypervolume: {ideal_hv:4f}")
    print(f"y_given Hypervolume: {y_given_hv:4f}")
    print(f"pf Hypervolume: {pf_hv:4f}")
    hv_results = {
        "Hypervolume (100th)": hv_value,
        "Hypervolume (D(best))": d_best_hv,
        "Ideal Hypervolume": ideal_hv,
        "y_given Hypervolume": y_given_hv,
        "pf Hypervolume": pf_hv,
    }
        # # 假设 pf 是你的帕累托前沿
    ind_gd = GD(pf)
    ind_gdp = GDPlus(pf)
    ind_igd = IGD(pf)
    ind_igdp = IGDPlus(pf)
    # 计算 MOO的各种 指标
    gd_plus_value = ind_gd.do(y_cand)
    gdp_plus_value = ind_gdp.do(y_cand)
    igd_value = ind_igd.do(y_cand)
    igd_plus_value = ind_igdp.do(y_cand)

    print("GD:", gd_plus_value)
    print("GD+:", gdp_plus_value)
    print("IGD:", igd_value)
    IGD_list.append(igd_value)
    print("IGD+:", igd_plus_value)
    indicators_results = {
        "GD": gd_plus_value,
        "GD+": gd_plus_value,
        "IGD": igd_value,
        "IGD+": igd_plus_value,
    }
    indicators_results.update(hv_results)
    # 假设y_cand和y_gen是两个NumPy数组
    # 计算l2距离
    l2_distance = np.linalg.norm(y_cand - y_given_list, ord=2)
    l2_distance1 = np.linalg.norm(y_gen - y_given_list, ord=2)

    print(f"L2 dist between design score and given y: {l2_distance}")
    print(f"L2 dist between generated and given y: {l2_distance1}")
    indicators_results['y_cand_y_given_l2_dist'] = l2_distance
    indicators_results['y_gen_y_given_l2_dist'] = l2_distance1

    spread_y_cand = calculate_spread(y_cand, ideal_points=ideal_point)
    spread_y_gen = calculate_spread(y_gen, ideal_points=ideal_point)
    print(f"Spread for y_cand: {spread_y_cand}")
    print(f"Spread for y_gen: {spread_y_gen}")  
    indicators_results['y_cand_spread'] = spread_y_cand
    indicators_results['y_gen_spread'] = spread_y_gen

    df = pd.DataFrame([indicators_results])
    filename = os.path.join(logging_dir, f"cond_{description}_all_indicators_results_shot{kkk}_seed{seed}.csv")
    df.to_csv(filename, index=False)

def get_best_result(task, res, xy_pred, y_given_list, clip_max, clip_min, kkk, config, augment=True, forward_model=None):
    #get best result
    x_size = task.x.shape[-1]
    y_size = task.y.shape[-1]
    input_size = x_size
    num_samples =  res['x'].shape[0] 
    valid_num = np.array([0]*len(y_given_list))  

    n = y_given_list.shape[0]
    min_dist = np.full(n, np.inf)  # 使用无穷大初始化，而不是1000
    x_opt_cand = np.zeros([n,x_size])  # 预先初始化x_opt_cand
    y_cand = np.zeros_like(y_given_list)  # 预先初始化y_cand
    y_gen = np.zeros_like(y_given_list)  # 预先初始化y_gen

    for i in range(num_samples):#num_samples

        clip_mtx =  np.array([1]*len(y_given_list))
        for idx, row in enumerate(xy_pred[f'{i}']):
            result_max, is_equal_max = check_elements(row, np.array(clip_max))
            result_min, is_equal_min = check_elements(row, np.array(clip_min))   
            if result_max or result_min:
                clip_mtx[idx] = 0
        valid_num += clip_mtx
        

        x_gen = xy_pred[f'{i}']
        x_cand = torch.tensor(x_gen[:,:input_size]).cpu().numpy()

        scores = task.predict(x_cand)#.ravel(order='F').reshape(num_samples, y_size) 

        if config['normalize_ys']:
            task.map_denormalize_y()
            scores = task.normalize_y(scores)

        if forward_model is not None:
            with torch.no_grad():
                y_pre = forward_model.forward(torch.tensor(x_gen).cuda()).cpu().numpy()
 
        if kkk==256:
            dist = np.linalg.norm(scores - y_given_list, axis=1)
        # dist = np.linalg.norm(scores_pre - y_given_list, axis=1)
        else:
            if augment:
                dist = np.linalg.norm(x_gen[:,input_size:]  - y_given_list , axis=1)
            else:
                dist = np.linalg.norm(y_pre - y_given_list , axis=1)

        #选择最好的
        mask = min_dist > dist
        if not np.any(np.isinf( scores[mask])):
            x_opt_cand[mask] = x_cand[mask]
            y_cand[mask] = scores[mask]
            if augment:
                y_gen[mask] =  x_gen[mask][:,input_size:]
            else:
                y_gen[mask] = y_pre[mask] 
            min_dist = np.minimum(min_dist, dist)
            
    return x_opt_cand, y_cand, y_gen, valid_num

from scipy.spatial.distance import cdist
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

def save_uncon_results(res, task, config, model, logging_dir, X, y, pf=None, augment=True):
    # 第一段代码的逻辑
    if pf is None:
        pf = task.problem.get_pareto_front()
        if config['normalize_ys']:
            if not task.problem.name.lower().startswith('zdt100'):
                pf = task.normalize_y(pf)
    y_given_list = pf  # [:100]
    x_size = task.x.shape[-1]
    y_size = task.y.shape[-1]
    input_size = x_size
    n = y_given_list.shape[0]
    k = 1  # 我们要找的最近节点数量
    y_cand = np.zeros((n, k, y_size))  # 预先初始化y_cand，假设y_size是y的维度
    min_dist = np.full((n, k), np.inf)  # 使用无穷大初始化最小距离数组
    if augment:
        x_gen = res['x']
        x_cand = torch.tensor(x_gen[:, :input_size]).cpu().numpy()
    else:
        x_gen =  res['x']
        x_cand = torch.tensor(x_gen).cpu().numpy()

    if 'DTLZ' in task.problem_name:
        scores = task.predict(x_cand).ravel(order='F').reshape(-1, y_size)
    else:
        scores = task.predict(x_cand)


    if config['normalize_ys']:
        scores = task.normalize_y(scores)

    # 计算所有scores和y_given_list之间的距离
    distances = cdist(y_given_list, scores)
    # distances = cdist(y_given_list, scores_pre)
    # distances = cdist(y_given_list, x_gen[:, input_size:])

    # 对于每个y_given_list中的点，找到距离最近点的索引
    nearest_indices = np.argmin(distances, axis=1)

    y_cand = scores[nearest_indices]
    if augment:
        y_gen = x_gen[:, input_size:][nearest_indices]
    else:
        with torch.no_grad():
            y_gen = model.forward(torch.tensor(x_gen).cuda()).cpu().numpy()[nearest_indices]
            
    # 假设y_cand和y_gen是两个NumPy数组
    # 计算l2距离
    l2_distance = np.linalg.norm(y_cand - y_given_list, ord=2)
    l2_distance1 = np.linalg.norm(y_gen - y_given_list, ord=2)

    nadir_point = task.nadir_point  # 手动在benchmark中已经设置好了
    ideal_point = task.problem.ideal_point
    if config['normalize_ys']:
        ideal_point = task.normalize_y(ideal_point)
        nadir_point = task.normalize_y(nadir_point)
    ideal_hv = hv(nadir_point, ideal_point, config['task'])
    hv_value = hv(nadir_point, y_cand, config['task'])

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Ideal Hypervolume: {ideal_hv:4f}")

    print(f"L2 dist between design score and given y: {l2_distance}")
    print(f"L2 dist between generated and given y: {l2_distance1}")

    # 第二段代码的逻辑
    res_x = res["x"]
    res_y = res["y_scores"]  # .ravel(order='F').reshape(num_samples, y_size)
    # task.predict(res_x[:,: task.x.shape[-1] ])
    x_size = task.x.shape[-1]
    y_size = task.y.shape[-1]
    # res_y = task.predict(res_x[:,: x_size ]).ravel(order='F').reshape(num_samples, y_size)
    # res_y = res["y"]
    # print(res_y)
    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point  # 手动在benchmark中已经设置好了
    ideal_point = task.problem.ideal_point
    if config['normalize_ys']:
        ideal_point = task.normalize_y(ideal_point)
        nadir_point = task.normalize_y(nadir_point)
        res_y = task.normalize_y(res_y)
        res_y_50_percent = task.normalize_y(res_y_50_percent)
        res_y_75_percent = task.normalize_y(res_y_75_percent)
 

    _, d_best = task.get_N_non_dominated_solutions(
        N= task.x.shape[0],  # int(len_data * config["data_preserved_ratio"]),#config["num_solutions"],
        return_x=True, return_y=True
    )

    best_sample = res_y.sum(axis=1).argmin()

    print('Best design is: \n', res_x[best_sample])
    print('Best design score is: \n', res_y[best_sample])
    best_sample_training = y.sum(axis=1).argmin()
    best_design_training = X[best_sample_training]
    # print(f"Best design in training set: {best_design_training}")
    print(f"Best design score in training set: \n {y[best_sample_training]}")
    print('Oracle design score is: \n', ideal_point)
    print('Distance between the best to oracle value in training set is :\n', np.linalg.norm(y[best_sample_training] - ideal_point))
    print('Distance between the best to oracle value is :\n', np.linalg.norm(res_y[best_sample] - ideal_point))

    d_best_hv = hv(nadir_point, d_best, config['task'])
    ideal_hv = hv(nadir_point, ideal_point, config['task'])
    hv_value = hv(nadir_point, res_y, config['task'])
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, config['task'])
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, config['task'])

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    print(f"Ideal Hypervolume: {ideal_hv:4f}")

    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/ideal": ideal_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
        "evaluation_step": 1000,
    }

    # # 假设 pf 是你的帕累托前沿
    ind_gd = GD(pf)
    ind_gdp = GDPlus(pf)
    ind_igd = IGD(pf)
    ind_igdp = IGDPlus(pf)

    # 计算 MOO的各种 指标
    gd_plus_value = ind_gd.do(res_y)
    gdp_plus_value = ind_gdp.do(res_y)
    igd_value = ind_igd.do(res_y)
    igd_plus_value = ind_igdp.do(res_y)

    print("GD:", gd_plus_value)
    print("GD+:", gdp_plus_value)
    print("IGD:", igd_value)
    print("IGD+:", igd_plus_value)
    indicators_results = {
        "GD": gd_plus_value,
        "GD+": gdp_plus_value,
        "IGD": igd_value,
        "IGD+": igd_plus_value,
    }
    indicators_results.update(hv_results)

    np.save(file=os.path.join(logging_dir, "uncond_sampling_res_x.npy"), arr=res_x)
    np.save(file=os.path.join(logging_dir, "uncond_sampling_res_y.npy"), arr=res_y)
    plot_y(res_y, pareto_front=pf, save_dir=logging_dir, config=config,
           nadir_point=nadir_point, d_best=y, ideal_point=ideal_point, description="uncond_sampling")

    df = pd.DataFrame([indicators_results])
    filename = os.path.join(logging_dir, "uncond_sampling_all_indicators_results.csv")
    df.to_csv(filename, index=False)

    return res_x, res_y, y_cand, y_gen, y_given_list



import matplotlib, math
from matplotlib import pyplot as plt
 

def plot_scatter(y_cand, y_gen, y_given_list, description, logging_dir, kkk, seed, augment=True):
    #plot scatterr
    n = y_cand.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(6*n, 5*n))
    fig.tight_layout(pad=3.6)
    scatter_size=50
    params = {
        'lines.linewidth': 2,
        'legend.fontsize': 22,
        'axes.labelsize': 28,
        'axes.titlesize': 36,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
    }
    matplotlib.rcParams.update(params)

    for y_i in range(n):
        for y_j in range(n):
            
            ax = axes[y_i, y_j]
            
            ax.scatter(y_cand[:, y_i], y_cand[:, y_j], color='b', label='Design Score', s=scatter_size,alpha=0.35)
            ax.scatter(y_gen[:, y_i], y_gen[:, y_j],color='g', label='Generated Score' if augment else 'Surrogated Score', s=scatter_size,alpha=0.5)
            if y_given_list.shape[0] > 200:
                ax.scatter(y_given_list[:, y_i], y_given_list[:, y_j], color='r', label='Pareto Front',alpha=0.02, s=scatter_size)
            else:
                ax.scatter(y_given_list[:, y_i], y_given_list[:, y_j], color='r', label='Expected Score',alpha=0.15, s=scatter_size)
            
            ax.set_xlabel(r'$y_{' + str(y_i) + '}$', fontdict={'family' : 'DejaVu Sans'})
            ax.set_ylabel(r'$y_{' + str(y_j) + '}$', fontdict={'family' : 'DejaVu Sans'})
            # 添加图例
            legend = ax.legend(fontsize=14)
            # 修改图例中标记的透明度为1
            for lh in legend.legendHandles:
                lh.set_alpha(1)
    if y_given_list.shape[0] > 2000:
        plt.savefig(f'{logging_dir}/uncond_pf_scatter.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig(f'{logging_dir}/cond_{description}_shot{kkk}_seed{seed}_pf_scatter.pdf', bbox_inches='tight', dpi=500)
    plt.show()
    
def plot_line(y_cand, y_gen, y_given_list, description, logging_dir, kkk, seed, augment=True):
    #plot line
    n = y_cand.shape[1]
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.tight_layout(pad=4.2)

    k = -1  # 保持原来的 k 值

    for y_i in range(n):
        row = y_i // cols
        col = y_i % cols
        
        ax = axes[row, col] if rows > 1 else axes[col]
        
        idx = np.argsort(y_given_list[:, y_i][:-1])
        
        ax.plot(y_given_list[:, y_i][:k][idx], y_cand[:, y_i][:k][idx], label='Design Score', color='b')
        ax.plot(y_given_list[:, y_i][:k][idx], y_gen[:, y_i][:k][idx], label='Generated Score' if augment else 'Surrogated Score', color='g')
        ax.plot(y_given_list[:, y_i][:k][idx], y_given_list[:, y_i][:k][idx], label='Expected Score', color='r',alpha=0.25)
        
        ax.set_xlabel(r'$y_{' + str(y_i) + '}$(Given)', fontdict={'family' : 'DejaVu Sans'})
        ax.set_ylabel(r'$y_{' + str(y_i) + '}$(Actual)', fontdict={'family' : 'DejaVu Sans'})
        # ax.legend( fontsize=14)
        legend = ax.legend(fontsize=14)
        # 修改图例中标记的透明度为1
        for lh in legend.legendHandles:
            lh.set_alpha(1)

    # 移除多余的子图
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    if y_given_list.shape[0] > 2000:
         plt.savefig(f'{logging_dir}/uncond_pf_line.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig(f'{logging_dir}/cond_{description}_shot{kkk}_seed{seed}_pf_line.pdf', bbox_inches='tight', dpi=500)
    plt.show()


def check_elements(array, clip):
    # 确保array和clip_min具有相同的形状
    
    # 比较array和clip_min
    is_equal = np.equal(array, clip)
    
    # 检查是否存在相等的元素
    has_equal_elements = np.any(is_equal)
    
    return has_equal_elements, is_equal

def calculate_spread(solutions, ideal_points=None):
    # 确保解决方案按照某个目标函数排序（例如，第一个目标）
    sorted_solutions = solutions[solutions[:, 0].argsort()]

    # 计算相邻解之间的欧氏距离
    distances = np.sqrt(np.sum(np.diff(sorted_solutions, axis=0)**2, axis=1))

    # 计算平均距离
    mean_distance = np.mean(distances)

    # 计算边界距离
    if ideal_points is None:
        # 使用解集中的极值点（假设第一个和最后一个解为边界解）
        d_extremes = np.sqrt(np.sum((sorted_solutions[[0, -1]] - 
                                     np.mean(sorted_solutions, axis=0))**2, axis=1))
    else:
        # 使用给定的理想点
        d_extremes = np.sqrt(np.sum((sorted_solutions[[0, -1]] - ideal_points)**2, axis=1))

    # 计算spread指标
    numerator = d_extremes.sum() + np.sum(np.abs(distances - mean_distance))
    denominator = d_extremes.sum() + len(distances) * mean_distance

    spread = numerator / denominator

    return spread


