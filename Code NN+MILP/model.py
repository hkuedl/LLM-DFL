
import torch
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB
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




class DeterministicLayer_milp:
    def __init__(self, args):
        """日前调度层初始化"""
        self.T = args.T
        self.N_g = args.N_g
        self.N = args.N
        
        # 转换参数为NumPy数组
        self.x_max = np.array(args.x_max, dtype=np.float64)
        self.x_min = np.array(args.x_min, dtype=np.float64)
        print(self.x_max)
        print(self.x_min)
        self.r_pos = np.array(args.r_pos, dtype=np.float64)
        self.r_neg = np.array(args.r_neg, dtype=np.float64)
        self.z_pos_max = np.array(args.z_pos_max, dtype=np.float64)
        self.z_neg_max = np.array(args.z_neg_max, dtype=np.float64)
        self.min_up_time = np.array(args.min_up_time, dtype=np.int32)  # 最小开机时间
        
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
        self.A_uq_x = np.vstack([row1, row2,row3])
        
        x_max_list=np.zeros(N_g * T, dtype=np.float64)
        for i in range(N_g):
             x_max_list[i*T:(i+1)*T] = self.x_max[i]*np.ones(T, dtype=np.float64)
    
    def solve(self, l_for, alpha, alpha_z):
        # 转换输入为NumPy数组并确保非负
        l_for = np.asarray(l_for, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        alpha_z = np.asarray(alpha_z, dtype=np.float64)
        
        # 创建Gurobi模型
        model = gp.Model("day_ahead_scheduling")
        model.setParam('OutputFlag', 0)  # 关闭输出
        
        # 添加决策变量
        x = model.addMVar(self.N_g * self.T, lb=0.0, name="x")
        z_max = model.addMVar(2 * self.N_g * self.T, lb=0.0, name="z_max")
        u = model.addMVar(self.N_g * self.T, vtype=GRB.BINARY, name="u")  # 机组启停状态
        
        # 原有约束
        model.addConstr(self.A_eq_l @ x == l_for, "power_balance")
        
        # 添加最小开机时间约束
        for g in range(self.N_g):
            min_up = self.min_up_time
            for t in range(self.T):
                end_time = min(t + min_up - 1, self.T - 1)
                for k in range(t, end_time + 1):
                    model.addConstr(u[g * self.T + k] >= u[g * self.T + t] - (1 if t > 0 else 0) * u[g * self.T + t - 1], 
                                  f"min_up_{g}_{t}_{k}")
                    
        for g in range(self.N_g):
            for t in range(self.T):
                idx = g * self.T + t
                model.addConstr(x[idx] <= self.x_max[g] * u[idx], f"on_off_upper_{g}_{t}")
                model.addConstr(x[idx] >= self.x_min[g] * u[idx], f"on_off_lower_{g}_{t}")
                model.addConstr(x[idx] + z_max[idx] <= self.x_max[g] * u[idx], f"non_negative_{g}_{t}")
                model.addConstr(x[idx] - z_max[self.N_g * self.T + idx] >= self.x_min[g] * u[idx], f"non_negative_{g}_{t}")
                model.addConstr(z_max[idx] <= self.z_pos_max * u[idx], f"z_pos_non_negative_{g}_{t}")
                model.addConstr(z_max[self.N_g * self.T + idx] <= self.z_neg_max * u[idx], f"z_neg_non_negative_{g}_{t}")
                if t > 0:
                    model.addConstr(x[idx]-x[idx-1] <= self.r_pos[g], f"ramp_up_{g}_{t}")
                    model.addConstr(x[idx-1]-x[idx] <= self.r_neg[g], f"ramp_down_{g}_{t}")

        obj = alpha @ x +  alpha_z @ z_max
        model.setObjective(obj, GRB.MINIMIZE)
        
        # 求解
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return x.X, z_max.X, u.X
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


class GurobiPowerScheduler_milp:
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
        self.deterministic_layer = DeterministicLayer_milp(args)
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
            x, z_max,_ = self.deterministic_layer.solve(forecasts_load, alpha, alpha_z)
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
