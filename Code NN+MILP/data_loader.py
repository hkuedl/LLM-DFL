from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Dataset_load_seq2seq(Dataset):#12264。15912 data(3 year)
    def __init__(self,  data_path='../Data/GEF_data/data(3 year).csv', flag='train', size=[96,0,24], train_length=15912, target='load', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        data=pd.read_csv(self.data_path)
        data = data.interpolate(method='cubic', limit_direction='both')

        for i in range(0,self.pred_len): 
            data[self.target+'+'+str(i)]=data['value'].shift(-i)
        for i in range(0,self.pred_len): 
            data['temp'+'+'+str(i)]=data['temp'].shift(-i)

        data.index=range(len(data))

        for i in range(self.seq_len):
            data['load_his_'+str(i+1)]=data['value'].shift(i+1)

        history_features = ['load_his_'+str(i+1) for i in range(self.seq_len)]
        T_features = ['temp'+'+'+str(i) for i in range(self.pred_len)]
        features_cal=['year','month','weekday']

        self.X_features_name = features_cal + history_features + T_features
        self.y_features_name = [self.target+'+'+str(i) for i in range(self.pred_len)]
        
        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)
        # 划分训练和测试数据
        print(np.shape(data))
        #train_data = data[0:24*(365+366-8-30)]
        #test_data = data[24*(365+366-8-30):24*(365+366-8)]
        train_data = data[0:24*(366+365-8-30)]
        test_data = data[24*(366+365-8-30):24*(366+365-9)]
        train_data.to_csv('../Data/GEF_data/train_data_2year.csv',index=False)
        test_data.to_csv('../Data/GEF_data/test_data_1month.csv',index=False)
        #test_data = data[24*(365+366):]
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集

        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)
            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)
            
            X_train_norm=X_train_norm[::24]
            y_train_norm=y_train_norm[::24]
            X_test_norm=X_test_norm[::24]
            y_test_norm=y_test_norm[::24]
            X_train=X_train_norm[0:366-8]
            y_train=y_train_norm[0:366-8]
            X_val, y_val = X_train_norm[366-8:], y_train_norm[366-8:]
            #
            # X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.3, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.3, random_state=42)
        
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        
        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)

    def inverse_transform_X(self, X_nor):
        return self.scaler_x.inverse_transform(X_nor)
    

class Dataset_load(Dataset):
    def __init__(self,  data_path='../Data/GEF_data/data.csv', flag='train', size=[96,0,24], train_length=8760, target='load', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        self.scaler_x_cal = StandardScaler() #MinMaxScaler()#
        data=pd.read_csv(self.data_path)
        data = data.interpolate(method='cubic', limit_direction='both')

        for j in range(int(self.seq_len/24)):
            data['load_'+str(j+1)+'_day_before']=data['value'].shift((j+1)*24)
        
        self.X_features_name = ['month','weekday','hour','temp'] + ['load_'+str(j+1)+'_day_before' for j in range(int(self.seq_len/24))]
        self.y_features_name = ['value']

        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)

        # 划分训练和测试数据
        train_data = data[0:self.train_length]
        test_data = data[self.train_length:]
        
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集
        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)

            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)

            X_train_norm=X_train_norm.reshape((-1,24,11))
            y_train_norm=y_train_norm.reshape(-1,24)
            X_test_norm=X_test_norm.reshape((-1,24,11))
            y_test_norm=y_test_norm.reshape(-1,24) 
            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
            
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        print(self.X.shape)
        print(self.y.shape)
        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)

    def inverse_transform_X(self, X_nor):
        return self.scaler_x.inverse_transform(X_nor)


import numpy as np

class CurveDictDB:
    def __init__(self):
        self.data = {}   # key: id, value: 记录字典
        self.curve_index = {}  # key: curve_key, value: list of ids

    def _curve_key(self, curve):
        # 使用 3 位小数防止浮点误差
        return tuple(round(x, 3) for x in curve)

    def insert(self, record):
        record_id = record['id']
        curve_key = self._curve_key(record['load'])

        if record_id in self.data:
            raise ValueError(f"id {record_id} 已存在，不能重复插入！")
        self.data[record_id] = record

        if curve_key not in self.curve_index:
            self.curve_index[curve_key] = []
        self.curve_index[curve_key].append(record_id)

    def get_top_k(self, curve_new, k=3, q=1, distance_type='euclidean', mode='good'):
        """
        查找相似负荷曲线 (不同于 curve_new)，分组后筛选并批量计算距离，返回前 k 条。
        """
        curve_new_key = self._curve_key(curve_new)
        grouped_records = {}

        # 基于索引和原始数据较快分组
        for curve_key, id_list in self.curve_index.items():
            if curve_key == curve_new_key:
                continue  # 跳过自身曲线
            records = [self.data[_id] for _id in id_list]
            grouped_records[curve_key] = []
            for record in records:
                # 保留你原来的筛选字段
                grouped_records[curve_key].append({
                    'id': record['id'],
                    'flag': record['flag'],
                    'prompt': record.get('prompt', ''),
                    'actual_load': record.get('actual_load', []),
                    'fine_tuned_load': record.get('fine_tuned_load', []),
                    'load': record['load'],
                    'cost': record['cost'],
                    'cost_reduction': record['cost_reduction']
                })

        # 模式筛选
        unique_records = []
        for curve_key, records in grouped_records.items():
            if mode == 'good':
                filtered_records = [r for r in records if r['cost_reduction'] >= 0]
                filtered_records.sort(key=lambda x: x['cost'])
            elif mode == 'bad':
                filtered_records = [r for r in records if r['cost_reduction'] < 0]
                filtered_records.sort(key=lambda x: x['cost'], reverse=True)
            else:
                raise ValueError(f"未知的mode: {mode}")
            unique_records.extend(filtered_records[:min(q, len(filtered_records))])

        if not unique_records:
            return []

        # 批量距离计算！
        loads_matrix = np.array([r['load'] for r in unique_records])
        curve_new_arr = np.array(curve_new)

        if distance_type == 'euclidean':
            dists = np.linalg.norm(loads_matrix - curve_new_arr, axis=1)
        elif distance_type == 'manhattan':
            dists = np.sum(np.abs(loads_matrix - curve_new_arr), axis=1)
        else:
            raise ValueError(f"未知的distance_type: {distance_type}")

        results = []
        for idx, record in enumerate(unique_records):
            results.append({
                'id': record['id'],
                'flag': record['flag'],
                'prompt': record.get('prompt', ''),
                'actual_load': record.get('actual_load', []),
                'fine_tuned_load': record.get('fine_tuned_load', []),
                'load': record['load'],
                'cost': record['cost'],
                'distance': dists[idx],
                'cost_reduction': record['cost_reduction']
            })

        results.sort(key=lambda x: x['distance'])
        return results[:min(k, len(results))]

    def get_top_k_same(self, curve_new, k=3, mode='good'):
        """
        查找与 curve_new 完全相同的负荷曲线记录，并按成本排序
        """
        curve_key = self._curve_key(curve_new)
        id_list = self.curve_index.get(curve_key, [])
        print(id_list)
        if not id_list:
            return []

        same_curve_records = [self.data[_id] for _id in id_list]
        base_record = next((r for r in same_curve_records if r['flag'] == 0), None)
        if base_record is None:
            return []

        base_cost = base_record['cost']
        filtered_records = []
        for record in same_curve_records:
            if mode == 'good' and record['cost'] < base_cost:
                filtered_records.append(record)
            elif mode == 'bad' and record['cost'] >= base_cost:
                filtered_records.append(record)

        filtered_records.sort(key=lambda x: x['cost'], reverse=(mode == 'bad'))
        return filtered_records[:k]

    def get_top_k_unique_loads(self, curve_new, k=3, q=1, distance_type='euclidean', mode='good'):
        """
        从 get_top_k 输出中选出负荷曲线不重复的前 k 条记录。
        """
        candidates = self.get_top_k(curve_new, k=max(20, k*2), q=q, distance_type=distance_type, mode=mode)
        unique_load_set = set()
        top_unique = []
        for r in candidates:
            load_key = tuple(round(x, 5) for x in r['load'])
            if load_key not in unique_load_set:
                unique_load_set.add(load_key)
                top_unique.append(r)
            if len(top_unique) == k:
                break
        return top_unique

    def rebuild_index(self):
        """从 self.data 重新扫描并生成 self.curve_index"""
        self.curve_index = {}
        for record_id, record in self.data.items():
            curve_key = self._curve_key(record['load'])
            if curve_key not in self.curve_index:
                self.curve_index[curve_key] = []
            self.curve_index[curve_key].append(record_id)


    def update(self, record_id, new_data):
        """
        更新指定 id 的实验记录
        """
        if record_id in self.data:
            self.data[record_id].update(new_data)
            return True
        return False

    def get_by_id(self, record_id):
        """
        根据 id 获取实验记录
        """
        return self.data.get(record_id)

    def all(self):
        """
        返回所有记录的列表
        """
        return list(self.data.values())