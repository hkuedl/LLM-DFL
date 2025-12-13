import torch
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import set_seed, MAPE, MAE, RMSE
import torch.nn as nn
import random
import copy
from data_loader import Dataset_load_seq2seq
from torch.utils.data import DataLoader


def E2E_train(args, combined_model, combined_fine_tune_loader, combined_test_loader, scaler_y_load, time_flag=False):
    device = args.device
    
    optimizer = torch.optim.SGD([
        {'params': combined_model.model_load.parameters()}
    ], lr=args.e2e_lr)

    plot_flag = False
    for i in range(3):
        print('===============================')
        print("e2e epoch:", i)
        print('===============================')
        for input_load, labels_load_nor, alpha, alpha_z, rho_z, rho_r in combined_fine_tune_loader:
            combined_model.train()
            input_load = input_load.to(device).float()
            labels_load_nor = labels_load_nor.to(device).float()
            alpha = alpha.to(device).float()
            alpha_z = alpha_z.to(device).float()
            rho_r = rho_r.to(device).float()    
            rho_z = rho_z.to(device).float()
            # ==== 嵌入 E2E_fine_tuning_parameter 的内容 ====
            optimizer.zero_grad()
            # reshape
            input_load = input_load.reshape(-1, input_load.shape[-1])
            # forward预测
            pred_mu, pred_var = combined_model.model_load_forward(input_load, scaler_y_load)

            #pred_load = combined_model.model_load_forward(input_load, scaler_y_load)
            # 数据归一化反变换
            if isinstance(scaler_y_load, StandardScaler):
                mean = torch.tensor(scaler_y_load.mean_, dtype=torch.float64).to(device)
                scale = torch.tensor(scaler_y_load.scale_,  dtype=torch.float64).to(device)
                labels_load = (labels_load_nor * scale + mean)
            # 判断 MinMaxScaler
            elif isinstance(scaler_y_load, MinMaxScaler):
                data_min = torch.tensor(scaler_y_load.data_min_,  dtype=torch.float64).to(device)
                data_max = torch.tensor(scaler_y_load.data_max_, dtype=torch.float64).to(device)
                labels_load = (labels_load_nor * (data_max - data_min) + data_min)
            # forward优化目标
            solution_ahead, solution_intra, obj = combined_model.forward(args, pred_mu,pred_var, labels_load, alpha, alpha_z, rho_z, rho_r)
            # loss & backward & optimizer
            loss = obj.mean()
            loss.backward()
            optimizer.step()


    return combined_model




def E2E_test(args, combined_model, combined_test_loader, scaler_y_load, time_flag=False):
    device = args.device
    cost_list = []
    solution_list_ahead = {}
    solution_list_intra = {}
    forecasts_load_mu = []
    forecasts_load_var = []
    actual_load = []
    input_return = []
    plot_flag = False  # 如有其他使用可改为参数

    # 用迭代器
    data_iter = iter(combined_test_loader)
    #next(data_iter)  # 跳过首批数据（如无需要可移除）

    while True:
        try:
            # 批量读取测试数据
            input_load_test, labels_load_test, alpha_test, alpha_z_test, rho_z_test, rho_r_test = next(data_iter)
        except StopIteration:
            print("No more data to read.")
            break

        # 数据准备
        input_load_test = input_load_test.to(device).float()
        labels_load_nor = labels_load_test.to(device).float()
        alpha_test = alpha_test.to(device).float()
        alpha_z_test = alpha_z_test.to(device).float()
        rho_z_test = rho_z_test.to(device).float()
        rho_r_test = rho_r_test.to(device).float()

        # ================== Evaluation_parameter_dynamic_price逻辑嵌入 ==================
        input_load_reshaped = input_load_test.reshape(-1, input_load_test.shape[-1])
        input_return.append(input_load_reshaped.detach().cpu())
        pred_mu, pred_var = combined_model.model_load_forward(input_load_reshaped, scaler_y_load)
        
        if isinstance(scaler_y_load, StandardScaler):
            scaler_load_mean = torch.tensor(scaler_y_load.mean_, dtype=torch.float64).to(device)
            scaler_load_scale = torch.tensor(scaler_y_load.scale_,  dtype=torch.float64).to(device)
            labels_load = (labels_load_nor * scaler_load_scale + scaler_load_mean)
        elif isinstance(scaler_y_load, MinMaxScaler):
            scaler_load_mean = torch.tensor(scaler_y_load.data_min_,  dtype=torch.float64).to(device)
            scaler_load_scale = torch.tensor(scaler_y_load.data_max_ - scaler_y_load.data_min_, dtype=torch.float64).to(device)
            labels_load = (labels_load_nor * (scaler_load_scale) + scaler_load_mean)


        set_seed(46)

        # 模型前向，获得目标与solution
        solution_ahead, solution_intra, obj = combined_model.forward(
            args, pred_mu, pred_var, labels_load, alpha_test, alpha_z_test, rho_z_test, rho_r_test
        )

        # solution字典拼接
        if not solution_list_ahead:
            solution_list_ahead = {k:v.detach().cpu() for k,v in solution_ahead.items()}
        else:
            for k in solution_ahead.keys():
                solution_list_ahead[k] = torch.cat([solution_list_ahead[k], solution_ahead[k].detach().cpu()])

        if not solution_list_intra:
            solution_list_intra = {k:v.detach().cpu() for k,v in solution_intra.items()}
        else:
            for k in solution_intra.keys():
                solution_list_intra[k] = torch.cat([solution_list_intra[k], solution_intra[k].detach().cpu()])

        print(f"Test Objective: {obj.mean().item()}")
        cost_list += list(obj.detach().cpu().numpy())

        forecasts_load_mu += list(pred_mu)
        forecasts_load_var += list(pred_var)
        actual_load += list(labels_load)
    input_return=torch.concat(input_return, dim=0)#.shape

    return solution_list_ahead, solution_list_intra, cost_list, forecasts_load_mu, forecasts_load_var, actual_load




def E2E_test_ideal(args, combined_model, combined_test_loader, scaler_y_load):
    device = args.device
    cost_list = []
    solution_list_ahead = {}
    solution_list_intra = {}
    forecasts_mu = []
    forecasts_var = []
    actual_load = []
    plot_flag = False  # 如有其他使用可改为参数

    # 用迭代器
    data_iter = iter(combined_test_loader)
    #next(data_iter)  # 跳过首批数据（如无需要可移除）

    while True:
        try:
            # 批量读取测试数据
            input_load_test, labels_load_test, alpha_test, alpha_z_test, rho_z_test, rho_r_test = next(data_iter)
        except StopIteration:
            print("No more data to read.")
            break

        # 数据准备
        input_load_test = input_load_test.to(device).float()
        labels_load_nor = labels_load_test.to(device).float()
        alpha_test = alpha_test.to(device).float()
        alpha_z_test = alpha_z_test.to(device).float()
        rho_z_test = rho_z_test.to(device).float()
        rho_r_test = rho_r_test.to(device).float()

        # ================== Evaluation_parameter_dynamic_price逻辑嵌入 ==================
        input_load_reshaped = input_load_test.reshape(-1, input_load_test.shape[-1])
        pred_mu, pred_var = combined_model.model_load_forward(input_load_reshaped, scaler_y_load)
        if isinstance(scaler_y_load, StandardScaler):
            scaler_load_mean = torch.tensor(scaler_y_load.mean_, dtype=torch.float64).to(device)
            scaler_load_scale = torch.tensor(scaler_y_load.scale_,  dtype=torch.float64).to(device)
            #pred_mu = pred_mu_nor * scaler_load_scale + scaler_load_mean
            #pred_var = pred_var_nor * scaler_load_scale
            labels_load = (labels_load_nor * scaler_load_scale + scaler_load_mean)
            labels_var = torch.zeros(np.shape(labels_load)).to(device)
            
        elif isinstance(scaler_y_load, MinMaxScaler):
            scaler_load_mean = torch.tensor(scaler_y_load.data_min_,  dtype=torch.float64).to(device)
            scaler_load_scale = torch.tensor(scaler_y_load.data_max_ - scaler_y_load.data_min_, dtype=torch.float64).to(device)
            #pred_mu = pred_mu_nor * (scaler_load_scale) + scaler_load_mean
            #pred_var = pred_var_nor * (scaler_load_scale)
            labels_load = (labels_load_nor * (scaler_load_scale) + scaler_load_mean)
            labels_var = torch.zeros(np.shape(labels_load)).to(device)

        set_seed(42)

        # 模型前向，获得目标与solution
        solution_ahead, solution_intra, obj = combined_model.forward(
            args, labels_load, labels_var, labels_load, alpha_test, alpha_z_test, rho_z_test, rho_r_test
        )
        # solution字典拼接
        if not solution_list_ahead:
            solution_list_ahead = {k:v.detach().cpu() for k,v in solution_ahead.items()}
        else:
            for k in solution_ahead.keys():
                solution_list_ahead[k] = torch.cat([solution_list_ahead[k], solution_ahead[k].detach().cpu()])

        if not solution_list_intra:
            solution_list_intra = {k:v.detach().cpu() for k,v in solution_intra.items()}
        else:
            for k in solution_intra.keys():
                solution_list_intra[k] = torch.cat([solution_list_intra[k], solution_intra[k].detach().cpu()])

        print(f"Test Objective: {obj.mean().item()}")
        cost_list += list(obj.detach().cpu().numpy())

        forecasts_mu += list(pred_mu.detach().cpu().numpy())
        forecasts_var += list(pred_var.detach().cpu().numpy())  
        actual_load += list(labels_load.detach().cpu().numpy())

    return solution_list_ahead, solution_list_intra, cost_list, forecasts_mu, forecasts_var, actual_load
