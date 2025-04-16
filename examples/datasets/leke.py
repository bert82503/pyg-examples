import os
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class LeKe(InMemoryDataset):
    r"""Taobao is a dataset of user behaviors from Taobao offered by Alibaba,
    provided by the `Tianchi Alicloud platform
    <https://tianchi.aliyun.com/dataset/649>`_.

    Taobao is a heterogeneous graph for recommendation.
    Nodes represent users with user IDs, items with item IDs, and categories
    with category ID.
    Edges between users and items represent different types of user behaviors
    towards items (alongside with timestamps).
    Edges between items and categories assign each item to its set of
    categories.

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

    """
    url = ('https://leoao-security-risk.oss-cn-hangzhou.aliyuncs.com/'
           'UserStoreBehavior0409.csv?Expires=1744349137&OSSAccessKeyId=TMP.3KsmdtPsaB4zev7hbTXFg9gMYtV1SU4SvpV3XWgz4sxL2pXaJTovAjREA3MFHm88iDSzHCpQ7LinXcDvmL9XTHWN5yC2uz&Signature=QhPfOQrAU6srDA16Jmw5N8QquE8%3D')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> str:
        # 原始的文件名称
        return 'UserStoreBehavior.csv'

    @property
    def processed_file_names(self) -> str:
        # 处理后的文件名称
        return 'data.pt'

    def download(self) -> None:
        # 下载文件url
        path = download_url(self.url, self.raw_dir)
        # 解压缩
        # extract_zip(path, self.raw_dir)
        # 移除压缩的文件
        # os.remove(path)

    def process(self) -> None:
        import pandas as pd

        # 读取csv文件数据
        # 列名
        # cols = ['userId', 'itemId', 'categoryId', 'behaviorType', 'timestamp']
        # user_id,store_id,city_id,behavior_type,behavior_time
        # user_id、city、store_id、behavior_type、behavior_time
        cols = ['userId', 'cityId', 'storeId', 'behaviorType', 'behaviorTime']
        df = pd.read_csv(self.raw_paths[0], names=cols)

        # Time representation (YYYY.MM.DD-HH:MM:SS -> Integer)
        # start: 1511539200 = 2017.11.25-00:00:00
        # end:   1512316799 = 2017.12.03-23:59:59
        # start = 1704038400
        # end = 1744100572
        # 时间戳过滤
        # df = df[(df["behaviorTime"] >= start) & (df["behaviorTime"] <= end)]

        # 删除重复项
        df = df.drop_duplicates()

        # 行为字典
        # behavior_dict = {'run': 1, 'door': 2, 'gc': 3, 'cp': 4, 'pr': 5}
        # df['behaviorType'] = df['behaviorType'].map(behavior_dict)

        # 实体数量
        num_entries = {}
        for name in ['userId', 'storeId', 'cityId']:
            # Map IDs to consecutive integers:
            # 将IDs映射到连续整数
            value, df[name] = np.unique(df[[name]].values, return_inverse=True)
            num_entries[name] = value.shape[0]
        print(f'num_entries={num_entries}')

        # 异构图的数据对象
        data = HeteroData()

        # 节点数量=实体数量
        # 用户、门店、城市
        data['user'].num_nodes = num_entries['userId']
        data['store'].num_nodes = num_entries['storeId']
        data['city'].num_nodes = num_entries['cityId']

        # <用户, 门店>边
        row = torch.from_numpy(df['userId'].values)
        col = torch.from_numpy(df['storeId'].values)
        # 边索引
        data['user', 'store'].edge_index = torch.stack([row, col], dim=0)
        # 边属性
        # 操作时间戳
        data['user', 'store'].time = torch.from_numpy(df['behaviorTime'].values)
        # 操作行为
        behavior = torch.from_numpy(df['behaviorType'].values)
        data['user', 'store'].behavior = behavior

        # 删除重复项
        df = df[['storeId', 'cityId']].drop_duplicates()
        # <门店, 城市>边
        row = torch.from_numpy(df['storeId'].values)
        col = torch.from_numpy(df['cityId'].values)
        # 边索引
        data['store', 'city'].edge_index = torch.stack([row, col], dim=0)

        # 预变换数据
        data = data if self.pre_transform is None else self.pre_transform(data)

        # 保存预处理后的数据
        self.save([data], self.processed_paths[0])
