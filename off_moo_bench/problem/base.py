import torch
import numpy as np
from pymoo.core.problem import Problem
import os

pareto_fronts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto_fronts")

class BaseProblem(Problem):
    def __init__(self, name, problem_type, n_obj, n_dim, requires_normalized_x=False,
                 xl=None, xu=None, nadir_point=None, ideal_point=None, **kwargs):
        super().__init__(
            n_var = n_dim, 
            n_obj = n_obj,
            xl = 0 if xl is None else xl,
            xu = 1 if xu is None else xu,
            **kwargs
        )
        self.name = name
        self.n_dim = n_dim
        self.problem_type = problem_type
        self.pareto_front = None
        self.requires_normalized_x = requires_normalized_x

        if isinstance(nadir_point, list):
            nadir_point = np.array(nadir_point)
        self.nadir_point = nadir_point
        if isinstance(ideal_point, list):
            ideal_point = np.array(ideal_point)
        self.ideal_point = ideal_point

        # if not hasattr(self, 'lbound'):
        #     if isinstance(self.xl, np.ndarray):
        #         self.lbound = torch.from_numpy(self.xl)
        #     else:
        #         self.lbound = torch.ones(self.n_obj).float() * self.xl

        # if not hasattr(self, 'ubound'):
        #     if isinstance(self.xu, np.ndarray):
        #         self.ubound = torch.from_numpy(self.xu)
        #     else:
        #         self.ubound = torch.ones(self.n_obj).float() * self.xu

    def generate_x(self, size, n_dim):
        raise NotImplementedError
    
    def get_nadir_point(self):
        raise NotImplementedError
    
    def get_ideal_point(self):
        raise NotImplementedError
    
    def get_pareto_front(self):
        if self.name == 'OmniTest':
            return self.evaluate(self._calc_pareto_set())
        elif self.name == 'VLMOP2':
            return self._calc_pareto_front()
        elif self.name.lower().startswith('zdt') :
            from pymoo.problems import get_problem
            return get_problem(self.name.lower()).pareto_front()
        elif  self.name.lower().startswith('dtlz'):
            from pymoo.problems import get_problem
            # print(f"self.name: {self.name}")
            if self.name.lower().startswith('dtlz5') or self.name.lower().startswith('dtlz7') or self.name.lower().startswith('dtlz6'):
                base_path = '/home/tzhouaq/offline-moo/off_moo_bench/problem/pareto_fronts/'#pareto_fronts_path
                pf_path = os.path.join(base_path, f"reference_points_{self.name}.npy")
                full_pf = np.load(pf_path)
                random_indices = np.random.choice(full_pf.shape[0], 500, replace=False)
                pf = full_pf[random_indices]
                return np.array(pf)
            else:
                return get_problem(self.name.lower()).pareto_front()
        elif self.name.lower().startswith('re'):
            base_path = pareto_fronts_path
            pf_path = os.path.join(base_path, f"reference_points_{self.name}.dat")
            assert os.path.exists(pf_path), f"Path of Pareto fronts of {self.name} not found: {pf_path}"

            with open(pf_path, "rb") as f:
                cot = f.read().decode('utf-8').split("\n")
                res = []
                for column in cot:
                    column = column.split()
                    if column:
                        res.append(list(map(float, column)))
            return np.array(res)
        elif self.name.lower().startswith('te'):
            base_path = pareto_fronts_path
            pf_path = os.path.join(base_path, f"reference_points_{self.name}.dat")
            print(f"pf_path: {pf_path}")
            assert os.path.exists(pf_path), f"Path of Pareto fronts of {self.name} not found: {pf_path}"

            with open(pf_path, "rb") as f:
                cot = f.read().decode('utf-8').split("\n")
                res = []
                for column in cot:
                    column = column.split()
                    if column:
                        res.append(list(map(float, column)))
            return np.array(res)


        elif self.pareto_front is not None:
            return self.pareto_front
        else:
            print(f"No get Pareto front")
            return None
        
        
    def _to_cuda(self, x):
        if x.device.type == 'cuda':
            if self.lbound is not None:
                self.lbound = self.lbound.cuda()
            if self.ubound is not None:
                self.ubound = self.ubound.cuda()

    def _evaluate(self, x):
        raise NotImplementedError
        