
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import MAPE, RMSE
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        
        # 添加输入层
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        
        # 添加隐藏层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # 添加输出层
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # 最后一层不使用激活函数
        return x



class Determinisitic_layer():
    def __init__(self, args):
        super(Determinisitic_layer, self).__init__()
        T = args.T
        N_g = args.N_g
        N = args.N
        l_for = cp.Parameter(T, nonneg=True)
        alpha = cp.Parameter(N_g * T, nonneg=True)
        alpha_z = cp.Parameter(2 * N_g * T, nonneg=True)
        x = cp.Variable(N_g * T, nonneg=True)
        z_max = cp.Variable(2 * N_g * T, nonneg=True)

        x_max = torch.tensor(args.x_max, dtype=torch.float64)
        r_pos = torch.tensor(args.r_pos, dtype=torch.float64)
        r_neg = torch.tensor(args.r_neg, dtype=torch.float64)

        I_N_g = torch.eye(N_g * T, dtype=torch.float64)
        I = torch.eye(T, dtype=torch.float64)
        Ramp_matrix = torch.zeros((T - 1, T), dtype=torch.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1
        Ramp_matrix_combined = torch.cat([
            torch.cat([Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix], dim=1)
        ], dim=0)
        I_combined = torch.cat([I, I, I], dim=1)

        A_eq_l = I_combined
       
        row1 = I_N_g#torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row2 = Ramp_matrix_combined#torch.cat([Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)
        row3 = -Ramp_matrix_combined#torch.cat([-Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)

        x_max_list = torch.zeros(N_g * T, dtype=torch.float64)
        r_pos_list = torch.zeros(N_g * (T-1), dtype=torch.float64)
        r_neg_list = torch.zeros(N_g * (T-1), dtype=torch.float64)
        for n in range(N_g):
            x_max_list[n * T:(n + 1) * T] = args.x_max[n]
            r_pos_list[n * (T-1):(n + 1) * (T-1)] = args.r_pos[n]
            r_neg_list[n * (T-1):(n + 1) * (T-1)] = args.r_neg[n]

        #r_pos_list = torch.tensor([r_pos for i in range(N_g * (T - 1))], dtype=torch.float64)
        #r_neg_list = torch.tensor([r_neg for i in range(N_g * (T - 1))], dtype=torch.float64)
        A_uq_x = torch.cat([row1, row2, row3], dim=0)
        b_uq_x = torch.cat([x_max_list, r_pos_list, r_neg_list])

        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g], dim=1)
        z_pos_max_list = []
        z_neg_maxlist = []

        row_1=I_N_g #torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row_2=torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row_3=torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), -I_N_g], dim=1)
        H_uq_x = torch.cat([row_1, -row_1], dim=0)
        H_uq_z = torch.cat([row_2, row_3], dim=0)
        h_uq = torch.cat([x_max_list, torch.tensor([0 for i in range(N_g * T)], dtype=torch.float64)])

        for n in range(N_g):
            z_pos_max_list += [args.z_pos_max[n] for i in range(T)]
            z_neg_maxlist += [args.z_neg_max[n] for i in range(T)]
        z_pos_max_list = torch.tensor(z_pos_max_list, dtype=torch.float64)
        z_neg_maxlist = torch.tensor(z_neg_maxlist, dtype=torch.float64)
        A_uq_z = torch.cat([row1, row2], dim=0)
        b_uq_z = torch.cat([z_pos_max_list, z_neg_maxlist])

        constraints = []
        constraints += [A_eq_l @ x == l_for]
        constraints += [A_uq_z @ z_max <= b_uq_z]
        constraints += [A_uq_x @ x <= b_uq_x]
        constraints += [
            H_uq_x @ x+ H_uq_z @ z_max <= h_uq
        ]

        objective = cp.Minimize(alpha @ x + alpha_z @ z_max)

        problem = cp.Problem(objective, constraints)
        self.cvxpyLayer = CvxpyLayer(problem, parameters=[l_for, alpha, alpha_z],
                                     variables=[x, z_max])



class Intra_schdule_layer():
    def __init__(self, args):
        super(Intra_schdule_layer, self).__init__()
        T = args.T
        N = args.N
        N_g = args.N_g

        l_for = cp.Parameter(T,nonneg = True)
        target_l = cp.Parameter(T,nonneg = True)
        x = cp.Parameter(N_g * T,nonneg = True)
        z_max = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_z = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_r = cp.Parameter(2*T,nonneg = True)
        
        error_l = l_for - target_l
        
        z = cp.Variable((2 * N_g * T), nonneg=True)
        r = cp.Variable(2*T,nonneg = True)
        
        I_N_g = torch.eye(N_g * T, dtype=torch.float64)
        I = torch.eye(T, dtype=torch.float64)
        Ramp_matrix = torch.zeros((T - 1, T), dtype=torch.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1

        I_combined = torch.hstack([I, I, I])
        F_eq_r = torch.hstack([I, -I])
        F_eq_z = torch.hstack([I_combined, -I_combined])
        G_eq_l = I
        h_eq = torch.zeros(T, dtype=torch.float64)
        row1 = torch.hstack([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)])
        row2 = torch.hstack([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g])
        F_uq = torch.vstack([row1, row2])

        constraints = []
        constraints += [
            F_uq @ z <= z_max,
            F_eq_z @ z + G_eq_l @ error_l + F_eq_r@ r == h_eq
        ]

        objective = cp.Minimize(rho_z @ z + rho_r @ r)
        
            # 定义并求解问题
        problem = cp.Problem(objective, constraints)
        
        self.cvxpyLayer = CvxpyLayer(problem, parameters=[z_max, l_for, target_l,rho_z,rho_r], 
                                        variables=[z, r])



class Combined_deterministic(torch.nn.Module):
    def __init__(self, Determinisitc_layer,Intra_schdule_layer, model_load):
        super(Combined_deterministic, self).__init__()
        self.model_load = model_load
        self.Deterministic_layer = Determinisitc_layer.cvxpyLayer
        self.Intra_schdule_layer = Intra_schdule_layer.cvxpyLayer

    def model_load_forward(self, x, scaler_y):
        if isinstance(scaler_y, StandardScaler):
            mean_torch = torch.tensor(scaler_y.mean_, dtype=torch.float64).to(x.device)
            scale_torch = torch.tensor(scaler_y.scale_, dtype=torch.float64).to(x.device)
            pred_load_nor = self.model_load(x)
            pred_load = pred_load_nor * scale_torch + mean_torch
            return pred_load
        
        elif isinstance(scaler_y, MinMaxScaler):
            min_torch = torch.tensor(scaler_y.data_min_, dtype=torch.float64).to(x.device)
            max_torch = torch.tensor(scaler_y.data_max_, dtype=torch.float64).to(x.device)
            pred_load_nor = self.model_load(x)
            pred_load = pred_load_nor*(max_torch - min_torch)+min_torch
        #print('pred_load_nor',pred_load_nor.shape,pred_load.shape)
        return pred_load
    
    def switch_solution_to_dict_ahead(self,solution):
        solution_dict={}
        variables_name=['x', 'z_max']
        
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict

    def switch_solution_to_dict_intra(self,solution):
        solution_dict={}
        variables_name=['z', 'r']
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict


    def forward(self,args,forecasts_load,targets_load,alpha,alpha_z,rho_z,rho_r):
        forecasts_load = forecasts_load
        alpha=torch.tensor(alpha,dtype=torch.float64).to(args.device)
        alpha_z=torch.tensor(alpha_z,dtype=torch.float64).to(args.device)
        rho_z=torch.tensor(rho_z,dtype=torch.float64).to(args.device)
        rho_r=torch.tensor(rho_r,dtype=torch.float64).to(args.device)
        
        while True:
            try:
                solution_ahead = self.Deterministic_layer(forecasts_load,alpha,alpha_z, solver_args={'solve_method':'ECOS'})
                solution_dict_ahead =self.switch_solution_to_dict_ahead(solution_ahead)
                break
            except Exception as e:
                solution_ahead = self.Deterministic_layer(forecasts_load,alpha,alpha_z, solver_args={'solve_method':'SCS','max_iters':5000})
                solution_dict_ahead =self.switch_solution_to_dict_ahead(solution_ahead)
                break
        while True:
            try:
                solution_intra=self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_load, targets_load,rho_z,rho_r, solver_args={'solve_method':'ECOS'})
                solution_dict_intra =self.switch_solution_to_dict_intra(solution_intra)
                break   
            except Exception as e:
                solution_intra=self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_load, targets_load,rho_z,rho_r, solver_args={'solve_method':'SCS','max_iters':5000})
                solution_dict_intra =self.switch_solution_to_dict_intra(solution_intra)
                break    
        obj = torch.sum(solution_dict_ahead['x'][:,0:args.N_g*args.T] * alpha, dim=1)+torch.sum(solution_dict_ahead['z_max'] * alpha_z, dim=1)\
            +torch.sum(solution_dict_intra['z'] * rho_z, dim=1)+torch.sum(solution_dict_intra['r'] * rho_r, dim=1)
        return solution_dict_ahead,solution_dict_intra, obj


import numpy as np
import gurobipy as gp
from gurobipy import GRB


class DeterministicLayer:
    def __init__(self, args):
        """日前调度层初始化"""
        self.T = args.T
        self.N_g = args.N_g
        self.N = args.N
        
        # 转换参数为NumPy数组
        self.x_max = np.array(args.x_max, dtype=np.float64)
        self.r_pos = np.array(args.r_pos, dtype=np.float64)
        self.r_neg = np.array(args.r_neg, dtype=np.float64)
        self.z_pos_max = np.array(args.z_pos_max, dtype=np.float64)
        self.z_neg_max = np.array(args.z_neg_max, dtype=np.float64)
        
        # 预计算约束矩阵
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        """预计算所有约束矩阵（完全对应CVXPY版本）"""
        T = self.T
        N_g = self.N_g
        
        # 单位矩阵
        I_N_g = np.eye(N_g * T, dtype=np.float64)
        I = np.eye(T, dtype=np.float64)
        
        # Ramp矩阵 (T-1) x T
        Ramp_matrix = np.zeros((T - 1, T), dtype=np.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1
        
        # Ramp_matrix_combined: 3(T-1) x 3T
        zero_block = np.zeros((T - 1, T), dtype=np.float64)
        Ramp_matrix_combined = np.vstack([
            np.hstack([Ramp_matrix, zero_block, zero_block]),
            np.hstack([zero_block, Ramp_matrix, zero_block]),
            np.hstack([zero_block, zero_block, Ramp_matrix])
        ])
        
        # I_combined: T x 3T
        I_combined = np.hstack([I, I, I])
        
        # A_eq_l (功率平衡约束矩阵)
        self.A_eq_l = I_combined
        
        # A_uq_x (发电机约束矩阵)
        row1 = I_N_g
        row2 = Ramp_matrix_combined
        row3 = -Ramp_matrix_combined
        self.A_uq_x = np.vstack([row1, row2, row3])
        

        x_max_list = np.zeros(N_g * T, dtype=np.float64)
        r_pos_list = np.zeros(N_g * (T-1), dtype=np.float64)
        r_neg_list = np.zeros(N_g * (T-1), dtype=np.float64)
        for n in range(N_g):
            x_max_list[n * T:(n + 1) * T] = self.x_max[n]
            r_pos_list[n * (T-1):(n + 1) * (T-1)] = self.r_pos[n]
            r_neg_list[n * (T-1):(n + 1) * (T-1)] = self.r_neg[n]

        print('x_max_list',x_max_list.shape)
        print('r_pos_list',r_pos_list.shape)
        print('r_neg_list',r_neg_list.shape)
        print(x_max_list)
       # x_max_list = np.array([self.x_max for i in range(N_g * T)], dtype=np.float64)  # 长度: N_g * T = 92
       # r_pos_list = np.array([self.r_pos for i in range(N_g * (T - 1))], dtype=np.float64)  # 长度: N_g * (T-1) = 69
       # r_neg_list = np.array([self.r_neg for i in range(N_g * (T - 1))], dtype=np.float64)  # 长度: N_g * (T-1) = 69
        
        self.b_uq_x = np.concatenate([x_max_list, r_pos_list, r_neg_list])
        
        # A_uq_z (备用约束矩阵)
        row1 = np.hstack([I_N_g, np.zeros((N_g * T, N_g * T), dtype=np.float64)])
        row2 = np.hstack([np.zeros((N_g * T, N_g * T), dtype=np.float64), I_N_g])
        self.A_uq_z = np.vstack([row1, row2])
        
        # b_uq_z (备用约束上限)
        # for n in range(N_g):
        #     z_pos_max_list += [args.z_pos_max[n] for i in range(T)]
        #     z_neg_maxlist += [args.z_neg_max[n] for i in range(T)]
        z_pos_max_list = np.repeat(self.z_pos_max, T)
        z_neg_max_list = np.repeat(self.z_neg_max, T)
        self.b_uq_z = np.concatenate([z_pos_max_list, z_neg_max_list])
        
        # H_uq_x, H_uq_z (混合约束矩阵)
        row_1 = I_N_g
        row_2 = np.hstack([I_N_g, np.zeros((N_g * T, N_g * T), dtype=np.float64)])
        row_3 = np.hstack([np.zeros((N_g * T, N_g * T), dtype=np.float64), -I_N_g])
        self.H_uq_x = np.vstack([row_1, -row_1])
        self.H_uq_z = np.vstack([row_2, row_3])
        
        # h_uq (混合约束上限)
        # h_uq = torch.cat([x_max_list, torch.tensor([0 for i in range(N_g * T)])])
        self.h_uq = np.concatenate([x_max_list, np.zeros(N_g * T, dtype=np.float64)])
        
        # x_max_list用于目标函数
        self.x_max_list = x_max_list
    
    def solve(self, l_for, alpha, alpha_z):
        # 转换输入为NumPy数组并确保非负
        l_for = np.asarray(l_for, dtype=np.float64)#.flatten()
        alpha = np.asarray(alpha, dtype=np.float64)#.flatten()
        alpha_z = np.asarray(alpha_z, dtype=np.float64)#.flatten()
        
        # 创建Gurobi模型
        model = gp.Model("day_ahead_scheduling")
        model.setParam('OutputFlag', 0)  # 关闭输出
        
        # 添加决策变量（对应CVXPY的nonneg=True）
        x = model.addMVar(self.N_g * self.T, lb=0.0, name="x")
        z_max = model.addMVar(2 * self.N_g * self.T, lb=0.0, name="z_max")
        
        model.addConstr(self.A_eq_l @ x == l_for, "power_balance")
        model.addConstr(self.A_uq_z @ z_max <= self.b_uq_z, "reserve_limit")
        model.addConstr(self.A_uq_x @ x <= self.b_uq_x, "generation_limit")
        model.addConstr(self.H_uq_x @ x + self.H_uq_z @ z_max <= self.h_uq, "combined_constraint")
        

        obj = alpha @ x + alpha_z @ z_max
        model.setObjective(obj, GRB.MINIMIZE)
        
        # 求解
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return x.X, z_max.X
        else:
            raise RuntimeError(f"日前调度优化失败，状态码: {model.status}")



class IntraScheduleLayer:
    def __init__(self, args):
        """日内调度层初始化"""
        self.T = args.T
        self.N = args.N
        self.N_g = args.N_g
        
        # 预计算约束矩阵
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        T = self.T
        N_g = self.N_g
        
        # 单位矩阵
        I_N_g = np.eye(N_g * T, dtype=np.float64)
        I = np.eye(T, dtype=np.float64)
        
        # I_combined: T x 3*N_g*T
        self.I_combined = np.hstack([I, I, I])
        
        # F_eq_r: T x 2T
        self.F_eq_r = np.hstack([I, -I])
        
        # F_eq_z: T x 2*N_g*T
        self.F_eq_z = np.hstack([self.I_combined, -self.I_combined])
        
        # G_eq_l: T x T
        self.G_eq_l = I
        
        # h_eq: T
        self.h_eq = np.zeros(T, dtype=np.float64)
        
        # F_uq: 2*N_g*T x 2*N_g*T
        row1 = np.hstack([I_N_g, np.zeros((N_g * T, N_g * T), dtype=np.float64)])
        row2 = np.hstack([np.zeros((N_g * T, N_g * T), dtype=np.float64), I_N_g])
        self.F_uq = np.vstack([row1, row2])
    
    def solve(self, z_max, l_for, target_l, rho_z, rho_r):
        # 转换输入为NumPy数组
        z_max = np.asarray(z_max, dtype=np.float64).flatten()
        l_for = np.asarray(l_for, dtype=np.float64).flatten()
        target_l = np.asarray(target_l, dtype=np.float64).flatten()
        rho_z = np.asarray(rho_z, dtype=np.float64).flatten()
        rho_r = np.asarray(rho_r, dtype=np.float64).flatten()
        
        # 计算负荷误差
        error_l = l_for - target_l
        
        # 创建Gurobi模型
        model = gp.Model("intra_day_scheduling")
        model.setParam('OutputFlag', 0)  # 关闭输出
        
        # 添加决策变量
        z = model.addMVar(2 * self.N_g * self.T, lb=0.0, name="z")
        r = model.addMVar(2 * self.T, lb=0.0, name="r")
        
        # 约束1: F_uq @ z <= z_max (备用容量限制)
        model.addConstr(self.F_uq @ z <= z_max, "reserve_limit")
        
        # 约束2: F_eq_z @ z + G_eq_l @ error_l + F_eq_r @ r == h_eq (功率平衡)
        model.addConstr(
            self.F_eq_z @ z + self.G_eq_l @ error_l + self.F_eq_r @ r == self.h_eq,
            "power_balance"
        )
        
        # 目标函数: rho_z @ z + rho_r @ r
        obj = rho_z @ z + rho_r @ r
        model.setObjective(obj, GRB.MINIMIZE)
        
        # 求解
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return z.X, r.X
        else:
            raise RuntimeError(f"日内调度优化失败，状态码: {model.status}")


class GurobiPowerScheduler:
    """Gurobi电力调度优化器（组合日前和日内调度）"""
    
    def __init__(self, args, alpha, alpha_z, rho_z, rho_r):
        """
        初始化调度器
        
        参数:
            args: 包含所有配置参数的对象
                - T: 时间段数
                - N_g: 发电机数量
                - N: 节点数量
                - x_max: 发电机最大功率
                - r_pos: 正向爬坡速率
                - r_neg: 负向爬坡速率
                - z_pos_max: 正向备用容量上限
                - z_neg_max: 负向备用容量上限
        """
        self.args = args
        self.deterministic_layer = DeterministicLayer(args)
        self.intra_schedule_layer = IntraScheduleLayer(args)
        self.alpha=alpha
        self.alpha_z=alpha_z
        self.rho_z=rho_z
        self.rho_r=rho_r
    
    def forward(self, forecasts_load, targets_load):
        # 转换输入为NumPy数组
        forecasts_load = np.asarray(forecasts_load, dtype=np.float64).flatten()
        targets_load = np.asarray(targets_load, dtype=np.float64).flatten()
        alpha = np.asarray(self.alpha, dtype=np.float64).flatten()
        alpha_z = np.asarray(self.alpha_z, dtype=np.float64).flatten()
        rho_z = np.asarray(self.rho_z, dtype=np.float64).flatten()
        rho_r = np.asarray(self.rho_r, dtype=np.float64).flatten()
        
        # 第一阶段：日前调度
        try:
            x, z_max = self.deterministic_layer.solve(forecasts_load, alpha, alpha_z)
        except Exception as e:
            print(f"日前调度求解失败: {e}")
            raise
        
        solution_ahead = {
            'x': x,
            'z_max': z_max
        }
        
        # 第二阶段：日内调度
        try:
            z, r = self.intra_schedule_layer.solve(z_max, forecasts_load, targets_load, rho_z, rho_r)
        except Exception as e:
            print(f"日内调度求解失败: {e}")
            raise
        
        solution_intra = {
            'z': z,
            'r': r
        }
        
        obj_ahead = np.sum(alpha * x) + np.sum(alpha_z * z_max)
        obj_intra = np.sum(rho_z * z) + np.sum(rho_r * r)
        obj = obj_ahead + obj_intra
        
        return solution_ahead, solution_intra, obj

