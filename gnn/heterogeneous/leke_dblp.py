import os.path as osp
from typing import Callable, List, Optional, Any

import math
import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
)

import pandas as pd
# from sklearn.preprocessing import StandardScaler

class LeKeDBLP(InMemoryDataset):
    r"""A subset of the DBLP computer science bibliography website, as
    collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    DBLP is a heterogeneous graph containing four types of entities - authors
    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
    (20 nodes).
    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Node/Edge Type
          - #nodes/#edges
          - #features
          - #classes
        * - Author
          - 4,057
          - 334
          - 4
        * - Paper
          - 14,328
          - 4,231
          -
        * - Term
          - 7,723
          - 50
          -
        * - Conference
          - 20
          - 0
          -
        * - Author-Paper
          - 196,425
          -
          -
        * - Paper-Term
          - 85,810
          -
          -
        * - Conference-Paper
          - 14,328
          -
          -
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        # self.user_scaler = StandardScaler()
        # self.device_scaler = StandardScaler()
        # self.ip_scaler = StandardScaler()

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'user_device0507.csv',
            'device_info0507.csv',
            'ip_info0507.csv',
            'user_identity0508.csv',
            'user_risk.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        print("\ndownload_file start")
        urls = [
            'https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/user_device0507.csv?Expires=1746610578&OSSAccessKeyId=TMP.3KpPoegwzP4W2GnxY4y9AJ5Vxo75wLxHrrRRCfJLgowxuDiDCG9nb64AAHzsdWA4gxo7ewRKnnCwdvpbwiC2yGbGMGyuXW&Signature=4o5KV7Kd5C6SfGw7Mkn0IRALvng%3D',
            'https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/device_info0507.csv?Expires=1746610595&OSSAccessKeyId=TMP.3KpPoegwzP4W2GnxY4y9AJ5Vxo75wLxHrrRRCfJLgowxuDiDCG9nb64AAHzsdWA4gxo7ewRKnnCwdvpbwiC2yGbGMGyuXW&Signature=YFB7l1T1axgc0IIOD9F1wZHJl08%3D',
            'https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/ip_info0507.csv?Expires=1746610613&OSSAccessKeyId=TMP.3KpPoegwzP4W2GnxY4y9AJ5Vxo75wLxHrrRRCfJLgowxuDiDCG9nb64AAHzsdWA4gxo7ewRKnnCwdvpbwiC2yGbGMGyuXW&Signature=U%2FHpdLcjeY96%2B8qUrQS%2FvZ1wiBU%3D',
            'https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/user_identity0508.csv?Expires=1746701335&OSSAccessKeyId=TMP.3KoiiqqxeMutZWFnJ4a5Js73wbYX7AFD1BtH568aNmzrnPHVoYXhXWuFb9fqgdjoaEnWRtmuNiZqToZuVRvEGSu48AeZHZ&Signature=yZa512JzLTJSDNd6d%2BvuD9VBi1E%3D',
            'https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/user_risk.csv?Expires=1746610682&OSSAccessKeyId=TMP.3KpPoegwzP4W2GnxY4y9AJ5Vxo75wLxHrrRRCfJLgowxuDiDCG9nb64AAHzsdWA4gxo7ewRKnnCwdvpbwiC2yGbGMGyuXW&Signature=uX0W6bHWjJyatZHyb68PMi%2F55J8%3D',
        ]
        for url in urls:
            print(f"download_file {url}")
            path = download_url(url, self.raw_dir)
        print("download_file done")
        # path = download_url(self.url, self.raw_dir)

    @staticmethod
    def check_tensor_data(log_name: Any, tensor_data: torch.Tensor) -> None:
        print(f"NaN in {log_name}:", torch.isnan(tensor_data).any())
        print(f"Inf in {log_name}:", torch.isinf(tensor_data).any())

    @staticmethod
    def check_series_data(log_name: Any, series_data: pd.Series) -> None:
        print(f"Null in {log_name}:", series_data.isnull().any())
        print(f"NaN in {log_name}:", series_data.isna().any())
        print(f"isin in {log_name}:", series_data.isin([None, '', 'null']).any())

    # 检查数据
    def check_data(self, user_df, device_df, ip_df, user_risk_df, user_device_ip_df) -> None:
        print("check_data start")

        # 用户ID重复
        user_df_size = len(user_df)
        user_ids = user_df['user_id'].unique()
        user_id_size = len(user_ids)
        print(f"user.size: {user_df_size}, user_ids.size: {user_id_size}")
        # 风险用户ID
        user_risk_df_size = len(user_risk_df)
        user_ids = user_risk_df['user_id'].unique()
        user_id_size = len(user_ids)
        print(f"user_risk.size: {user_risk_df_size}, user_ids.size: {user_id_size}")

        print("user")
        # 检查NaN/Inf
        user_id_tensor = torch.tensor(user_df['user_id'], dtype=torch.long)
        self.check_tensor_data('user_id_tensor', user_id_tensor)
        identity_tensor = torch.tensor(user_df['identity'], dtype=torch.float)
        self.check_tensor_data('identity_tensor', identity_tensor)
        risk_score_tensor = torch.tensor(user_df['risk_score'], dtype=torch.float)
        self.check_tensor_data('risk_score_tensor', risk_score_tensor)

        print("device")
        # 值是字符串
        self.check_series_data('device_id_series', device_df['device_id'])

        print("ip")
        # 值是字符串
        self.check_series_data('ip_series', ip_df['ip'])

        print("user_risk")
        # 检查NaN/Inf
        user_id_tensor = torch.tensor(user_risk_df['user_id'], dtype=torch.long)
        self.check_tensor_data('user_id_tensor', user_id_tensor)

        print("user_device_ip")
        # 检查NaN/Inf
        user_id_tensor = torch.tensor(user_device_ip_df['user_id'], dtype=torch.long)
        self.check_tensor_data('user_id_tensor', user_id_tensor)
        timestamp_tensor = torch.tensor(user_device_ip_df['timestamp'], dtype=torch.long)
        self.check_tensor_data('timestamp_tensor', timestamp_tensor)
        # 值是字符串
        self.check_series_data('device_id_series', user_device_ip_df['device_id'])
        self.check_series_data('ip_series', user_device_ip_df['ip'])

        print("check_data done")

    # 加载数据
    def load_data(self):
        print("\nload_data start")

        data_dir = osp.join(self.root, 'raw')
        print(f"data_dir: {data_dir}")

        file_names = self.raw_file_names

        # 设备数据
        device_df_path = osp.join(data_dir, file_names[1])
        device_df_cols = ['device_id']
        device_df = pd.read_csv(device_df_path, names=device_df_cols)
        device_df['used_times'] = 1
        print("device_df")

        # IP数据
        ip_df_path = osp.join(data_dir, file_names[2])
        ip_df_cols = ['ip']
        ip_df = pd.read_csv(ip_df_path, names=ip_df_cols)
        ip_df['used_times'] = 1
        print("ip_df")

        # 用户数据
        user_df_path = osp.join(data_dir, file_names[3])
        user_df_cols = ['user_id', 'identity', 'risk_score']
        user_df = pd.read_csv(user_df_path, names=user_df_cols)
        print("user_df")

        # 风险用户数据
        user_risk_df_path = osp.join(data_dir, file_names[4])
        user_risk_df_cols = ['user_id']
        user_risk_df = pd.read_csv(user_risk_df_path, names=user_risk_df_cols)
        print("user_risk_df")

        # 风险用户-标签
        user_df['risk_type'] = 0
        risk_user_ids = user_risk_df['user_id']
        user_df.loc[user_df['user_id'].isin(risk_user_ids), 'risk_type'] = 1
        print("user_risk_type")

        # 用户设备IP数据
        user_device_ip_df_path = osp.join(data_dir, file_names[0])
        user_device_ip_df_cols = ['user_id', 'device_id', 'timestamp', 'ip']
        user_device_ip_df = pd.read_csv(user_device_ip_df_path, names=user_device_ip_df_cols)
        print("user_device_ip_df")

        self.check_data(user_df, device_df, ip_df, user_risk_df, user_device_ip_df)

        print("load_data done")
        return user_df, device_df, ip_df, user_device_ip_df

    # 构建索引下标映射
    @staticmethod
    def build_index_map(user_df, device_df, ip_df):
        print("build_index_map start")

        """预构建索引映射，O(1)时间复杂度查找"""
        # 用户ID到索引的哈希映射
        user_idx_map = pd.Series(user_df.index, index=user_df['user_id']).to_dict()
        # 设备ID到索引的映射
        device_idx_map = pd.Series(device_df.index, index=device_df['device_id']).to_dict()
        # device_ids = device_df['device_id']
        # # <device_id, index>
        # device_idx_map = {device_id: idx for idx, device_id in enumerate(device_ids)}
        # IP到索引的映射
        ip_idx_map = pd.Series(ip_df.index, index=ip_df['ip']).to_dict()
        # ips = ip_df['ip']
        # # <ip, index>
        # ip_idx_map = {ip: idx for idx, ip in enumerate(ips)}

        print("build_index_map done")
        return user_idx_map, device_idx_map, ip_idx_map

    # 构建边
    def build_edges(self, user_df, device_df, ip_df, user_device_ip_df):
        print("\nbuild_edges start")
        # 节点索引
        user_idx_map, device_idx_map, ip_idx_map = self.build_index_map(user_df, device_df, ip_df)

        # 构建边关系
        edges = []
        # 用户-设备边
        user_use_device_map = [[], []]
        # 设备-IP边
        device_connect_ip_map = [[], []]
        # 用户-IP边
        user_access_ip_map = [[], []]
        for _idx, row in user_device_ip_df.iterrows():
            # 用户数据行下标
            user_idx = user_idx_map[row['user_id']]
            # 设备数据行下标
            device_idx = device_idx_map[row['device_id']]
            # IP数据行下标
            ip_idx = ip_idx_map[row['ip']]
            # 边构建
            user_use_device_map[0].append(user_idx)
            user_use_device_map[1].append(device_idx)
            device_connect_ip_map[0].append(device_idx)
            device_connect_ip_map[1].append(ip_idx)
            user_access_ip_map[0].append(user_idx)
            user_access_ip_map[1].append(ip_idx)
        # 用户使用设备的边
        edges.append(('user', 'use', 'device', user_use_device_map[0], user_use_device_map[1]))
        # 反向边：device -> user
        edges.append(('device', 'used_by', 'user', user_use_device_map[1], user_use_device_map[0]))
        # 设备连接IP的边
        edges.append(('device', 'connect', 'ip', device_connect_ip_map[0], device_connect_ip_map[1]))
        # 反向边：ip -> device
        edges.append(('ip', 'connected_by', 'device', device_connect_ip_map[1], device_connect_ip_map[0]))
        # 用户连接IP的边
        edges.append(('user', 'access', 'ip', user_access_ip_map[0], user_access_ip_map[1]))
        # 反向边：ip -> user
        edges.append(('ip', 'accessed_by', 'user', user_access_ip_map[1], user_access_ip_map[0]))

        print("build_edges done")
        return edges

    def process(self) -> None:
        user_df, device_df, ip_df, user_device_ip_df = self.load_data()
        edges = self.build_edges(user_df, device_df, ip_df, user_device_ip_df)

        # # 特征标准化
        # user_df[['risk_score']] = self.user_scaler.fit_transform(
        #     user_df[['risk_score']])
        # device_df[['used_times']] = self.device_scaler.fit_transform(
        #     device_df[['used_times']])
        # ip_df[['used_times']] = self.ip_scaler.fit_transform(
        #     ip_df[['used_times']])
        # 1.0->0.0

        print("\nprocess start")
        data = HeteroData()

        # 节点特征
        data['user'].x = torch.tensor(
            user_df['risk_score'].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1)
        data['device'].x = torch.tensor(
            device_df['used_times'].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1)
        data['ip'].x = torch.tensor(
            ip_df['used_times'].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1)
        # data['device'].x = torch.ones([len(device_df['device_id']), 1])
        # data['ip'].x = torch.ones([len(ip_df['ip']), 1])
        # 节点数量
        data['user'].num_nodes = user_df.shape[0]
        data['device'].num_nodes = device_df.shape[0]
        data['ip'].num_nodes = ip_df.shape[0]

        # 验证节点特征维度
        # 检查用户、设备、IP节点的特征维度
        print("用户节点特征维度:", data['user'].x.shape)  # 应为 (num_users, 1)
        print("设备节点特征维度:", data['device'].x.shape)  # 应为 (num_devices, 1)
        print("IP节点特征维度:", data['ip'].x.shape)  # 应为 (num_ips, 1)

        # 标签
        data['user'].y = torch.tensor(
            user_df['risk_type'].values.astype(np.longlong),
            dtype=torch.long
        )

        # split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        # for name in ['train', 'val', 'test']:
        #     idx = split[f'{name}_idx']
        #     idx = torch.from_numpy(idx).to(torch.long)
        #     mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
        #     mask[idx] = True
        #     data['author'][f'{name}_mask'] = mask

        # user_size = data['user'].x.shape[0]
        user_size = data['user'].num_nodes
        # 训练集、验证集、测试集的数量
        # num_val = math.ceil(user_size * 0.1)
        # num_test = math.ceil(user_size * 0.2)
        num_val = math.ceil(user_size * 0.05)
        num_test = math.ceil(user_size * 0.1)
        num_train = user_size - num_val - num_test

        # data = self.get(0)
        train_mask = torch.zeros(user_size, dtype=torch.bool)
        train_mask[:num_train] = True
        data['user'].train_mask = train_mask

        val_mask = torch.zeros(user_size, dtype=torch.bool)
        val_mask[num_train:num_train + num_val] = True
        data['user'].val_mask = val_mask

        test_mask = torch.zeros(user_size, dtype=torch.bool)
        test_mask[num_train + num_val:] = True
        data['user'].test_mask = test_mask

        # 边索引和属性
        edge_types = {
            ('user', 'use', 'device'): {'index': []},
            ('device', 'used_by', 'user'): {'index': []},
            ('device', 'connect', 'ip'): {'index': []},
            ('ip', 'connected_by', 'device'): {'index': []},
            ('user', 'access', 'ip'): {'index': []},
            ('ip', 'accessed_by', 'user'): {'index': []},
        }

        for edge in edges:
            src_type, rel_type, dst_type, src, dst = edge
            edge_types[(src_type, rel_type, dst_type)]['index'].append([src, dst])

        for edge_type, data_dict in edge_types.items():
            src_type, rel_type, dst_type = edge_type
            edge_index = torch.tensor(data_dict['index'][0], dtype=torch.long)
            # 边索引
            data[edge_type].edge_index = edge_index
            # data[src_type, dst_type].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

        print("process done")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
