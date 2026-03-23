import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import copy
from data_loader import Dataset_load,Dataset_load_seq2seq
import matplotlib.pyplot as plt
import numpy as np
from http import HTTPStatus
import dashscope
from dashscope import Generation
import json
import os
from scipy.stats import norm

plt.rcParams["font.family"] = "Times New Roman"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def PICP(y, y_lower, y_upper):
    return np.mean((y >= y_lower) & (y <= y_upper))

def MAPE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    MAPE=np.mean(abs((y_actual-y_predicted)/y_actual))
    return MAPE

def R2(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    R2 = 1 - np.sum(np.square(y_actual-y_predicted)) / np.sum(np.square(y_actual-np.mean(y_actual)))
    return R2

def RMSE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    RMSE = np.sqrt(np.mean(np.square(y_actual-y_predicted)))
    return RMSE

def MAE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    return np.mean(np.abs(y_actual-y_predicted))

def get_load_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    if args.mode == 'seq2seq': 
        data_set=Dataset_load_seq2seq(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    else:
        data_set=Dataset_load(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader



def get_combined_data(args,load_data,alpha,alpha_z,rho_z,rho_r,flag='train'):
    load_data_X = torch.tensor(load_data.X, dtype=torch.float64)
    load_data_y = torch.tensor(load_data.y, dtype=torch.float64)
    alpha = torch.tensor(alpha, dtype=torch.float64)
    alpha_z = torch.tensor(alpha_z, dtype=torch.float64)
    
    rho_z = torch.tensor(rho_z, dtype=torch.float64)
    rho_r = torch.tensor(rho_r, dtype=torch.float64)
    batch_size = args.batch_size
    # 创建 TensorDataset
    combined_train_data = TensorDataset(load_data_X, load_data_y,alpha,alpha_z,rho_z,rho_r)
    shuffle_flag=True
    if flag=='test':
        shuffle_flag=False
        print('Test data is not shuffled')
        batch_size = 7
    combined_train_loader=DataLoader(combined_train_data, batch_size=batch_size, shuffle=shuffle_flag)
    return combined_train_data,combined_train_loader



def obtain_price(args,train_load_data,val_load_data,test_load_data):
    if args.flag_dynamic_price:
        if args.flag_dynamic_mode==1:
            alpha_list_train = [args.alpha for j in range(np.shape(train_load_data.X)[0])]  
            alpha_z_list_train = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_load_data.X)[0])]
            rho_z_list_train = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_load_data.X)[0])]
            rho_r_list_train = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_load_data.X)[0])]

            alpha_list_val = [args.alpha for j in range(np.shape(val_load_data.X)[0])]
            alpha_z_list_val = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_z_list_val = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_r_list_val = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_pv_data.X)[0])]

            alpha_list_test = [args.alpha for j in range(np.shape(test_load_data.X)[0])]
            alpha_z_list_test = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_z_list_test = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_r_list_test = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_pv_data.X)[0])]

        else:
            alpha_list_train = [args.alpha for j in range(np.shape(train_load_data.X)[0])]  
            alpha_z_list_train = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_z_list_train = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_r_list_train = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_pv_data.X)[0])]

            alpha_list_val = [args.alpha for j in range(np.shape(val_load_data.X)[0])]
            alpha_z_list_val = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_z_list_val = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_r_list_val = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_pv_data.X)[0])]

            alpha_list_test = [args.alpha for j in range(np.shape(test_load_data.X)[0])]
            alpha_z_list_test = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_z_list_test = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_r_list_test = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_pv_data.X)[0])]
    
    else:
        if args.flag_dynamic_mode==1:
            ratio=args.price_ratio_small
        else:
            ratio=args.price_ratio_large
        
        alpha_list_train = [args.alpha for j in range(np.shape(train_load_data.X)[0])]
        alpha_z_list_train = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_load_data.X)[0])]
        rho_z_list_train = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_load_data.X)[0])]
        rho_r_list_train = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_load_data.X)[0])]

        alpha_list_val = [args.alpha for j in range(np.shape(val_load_data.X)[0])]
        alpha_z_list_val = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_load_data.X)[0])]
        rho_z_list_val = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_load_data.X)[0])]
        rho_r_list_val = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_load_data.X)[0])]

        alpha_list_test = [args.alpha for j in range(np.shape(test_load_data.X)[0])]
        alpha_z_list_test = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_load_data.X)[0])]
        rho_z_list_test = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_load_data.X)[0])]
        rho_r_list_test = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_load_data.X)[0])]

    combined_train_data,combined_train_loader=get_combined_data(args,train_load_data,
                        alpha_list_train,alpha_z_list_train,rho_z_list_train,rho_r_list_train,flag='train')

    
    combined_val_data,combined_val_loader=get_combined_data(args,val_load_data,
                        alpha_list_val,alpha_z_list_val,rho_z_list_val,rho_r_list_val,flag='val')
                                                            
    combined_test_data,combined_test_loader=get_combined_data(args,test_load_data,
                        alpha_list_test,alpha_z_list_test,rho_z_list_test,rho_r_list_test,flag='test')

    combined_fine_tune_data,combined_fine_tune_loader=get_combined_data(args,test_load_data,
                        alpha_list_test,alpha_z_list_test,rho_z_list_test,rho_r_list_test,flag='test')

    return combined_train_data,combined_train_loader,combined_val_data,combined_val_loader,combined_test_data,combined_test_loader,combined_fine_tune_data,combined_fine_tune_loader

import re
import inspect
import numpy as np

def evaluate_individual_strategies(func_code, original_curve, max_change_percent=0.03):
    """
    执行包含多个策略函数的代码，并分别评估每个策略
    
    参数:
        func_code (str): 包含多个策略函数定义的代码字符串
        original_curve (list): 原始负荷曲线 (24小时值)
        max_change_percent (float): 每个时间点最大调整百分比
        
    返回:
        dict: 键为策略函数名，值为字典包含:
            'adjusted_curve': 调整后的曲线
            'delta_curve': delta值列表
            'func_code': 该策略函数的完整代码
    """
    # 创建执行环境
    exec_env = {}
    try:
        # 执行整个代码块，注册所有函数
        exec(func_code, exec_env)
    except Exception as e:
        print(f"执行策略代码块失败: {str(e)}")
        return {}
    
    # 收集所有以'strategy_'开头的函数
    strategy_funcs = {}
    for name, obj in exec_env.items():
        if name.startswith('strategy_') and callable(obj):
            strategy_funcs[name] = obj
    
    if not strategy_funcs:
        print("未找到策略函数（函数名应以'strategy_'开头）")
        return {}
    
    # 对每个策略单独执行
    results = {}
    for name, func in strategy_funcs.items():
        try:
            # 执行策略函数获取deltas
            deltas = func(original_curve)
            if len(deltas) != 24:
                print(f"策略 {name} 返回的delta列表长度错误: {len(deltas)} != 24")
                continue
                
            # 应用调整（考虑最大调整限制）
            adjusted_curve = []
            for i in range(24):
                change = deltas[i]
                max_change = original_curve[i] * max_change_percent
                if abs(change) > max_change:
                    change = np.sign(change) * max_change
                adjusted_curve.append(original_curve[i] + change)
            
            # 改进的函数代码提取方法
            # 方法1: 使用AST解析整个代码块
            try:
                import ast
                tree = ast.parse(func_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == name:
                        # 获取函数定义的起始和结束行号
                        start_line = node.lineno
                        end_line = node.end_lineno
                        lines = func_code.split('\n')
                        func_str = '\n'.join(lines[start_line-1:end_line])
                        break
                else:
                    func_str = f"# 无法提取函数 {name} 的完整代码"
            except Exception:
                # 方法2: 使用改进的正则表达式捕获完整函数体
                try:
                    # 匹配从def开始到下一个def或文件结束
                    pattern = rf'(def\s+{name}\(.*?\):.*?)(?=\n\s*def\s|\Z)'
                    match = re.search(pattern, func_code, re.DOTALL)
                    if match:
                        func_str = match.group(1).strip()
                    else:
                        # 方法3: 回退到简单定义
                        func_str = f"def {name}(original_load):\n    # 函数代码提取不完整\n    pass"
                except Exception:
                    func_str = f"def {name}(original_load):\n    # 函数代码提取失败\n    pass"
            
            results[name] = {
                'adjusted_curve': adjusted_curve,
                'delta_curve': deltas,
                'func_code': func_str
            }
        except Exception as e:
            print(f"执行策略 {name} 时出错: {str(e)}")
    
    return results


def execute_strategy_function(func_code, original_load, max_change_percent=0.05):
    """
    安全执行模型生成的策略函数
    """
    # 创建安全执行环境 - 添加 enumerate 和 zip
    restricted_globals = {
        '__builtins__': {
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'len': len,
            'range': range,
            'enumerate': enumerate,  # 添加 enumerate
            'zip': zip,              # 添加 zip
            'list': list,            # 添加 list
            'print': print
            # 仅允许安全的数学函数
        },
        'original_load': original_load.copy(),
        'np': None  # 禁止numpy
    }
    
    # 定义策略函数名称
    function_name = "adjustment_strategy"
    
    # 编译和执行函数
    try:
        # 编译函数代码
        compiled = compile(func_code, '<string>', 'exec')
        
        # 在受限环境中执行
        exec(compiled, restricted_globals)
        
        # 获取函数对象
        strategy_func = restricted_globals.get(function_name)
        
        if not callable(strategy_func):
            raise ValueError(f"Function '{function_name}' not found or not callable")
        
        # 执行策略函数
        delta_curve = strategy_func(original_load)
        
        # 验证输出
        if not isinstance(delta_curve, list) or len(delta_curve) != 24:
            raise ValueError("Delta curve must be a list of 24 numbers")
        
        # 应用变化量约束
        adjusted_curve = []
        for i in range(24):
            max_delta = original_load[i] * max_change_percent
            adjusted_value = original_load[i] + max(-max_delta, min(delta_curve[i], max_delta))
            adjusted_curve.append(adjusted_value)
            
        return adjusted_curve, delta_curve, func_code  # 返回原始代码
    
    except Exception as e:
        print(f"策略函数执行错误: {str(e)}")
        # 返回原始曲线和原始代码（即使执行失败也保留代码）
        return original_load, [0]*24, func_code


def extract_pure_code(response_text):
    """
    从模型响应中提取纯Python代码
    """
    # 尝试匹配代码块
    import re
    code_pattern = r'```python(.*?)```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        # 返回第一个匹配的代码块
        return matches[0].strip()
    
    # 如果没有代码块，尝试查找函数定义
    func_pattern = r'def adjustment_strategy\(.*?\):.*?return'
    if re.search(func_pattern, response_text, re.DOTALL):
        # 提取从函数定义开始到最后一个return的代码
        start_idx = response_text.find('def adjustment_strategy')
        return response_text[start_idx:]
    
    # 如果都没有，返回原始文本
    return response_text


def get_fail_reflections_by_curve_distance(reflection_records, current_load, max_num=2):
    """
    根据和当前曲线的距离，选最近 max_num 个失败案例
    """
    # 只考虑 cost_after > cost_before 的反思
    fail_candidates = [r for r in reflection_records if r['cost_after'] > r['cost_before']]
    dist_list = []
    for r in fail_candidates:
        curve = np.array(r['original_load']).flatten()
        dist = np.linalg.norm(np.array(current_load).flatten() - curve)
        dist_list.append((dist, r))
    # 按距离升序排序
    dist_list.sort(key=lambda x: x[0])
    # 取距离最近的 max_num 个
    nearest_fail_records = [r for (_, r) in dist_list[:max_num]]
    return nearest_fail_records