# An Multi GPU implementation of unsupervised bipartite GraphSAGE
# using the LeKe dataset.
import argparse
import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import Embedding, Linear
from torch.nn.parallel import DistributedDataParallel

import torch_geometric.transforms as T
from examples.datasets.leke import LeKe
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.sampler import NegativeSampling
from torch_geometric.sampler.base import NegativeSamplingMode
from torch_geometric.utils.convert import to_scipy_sparse_matrix


# 商品编码
class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 二层隐藏层
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # 一层线性变换
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 边
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        # 一层线性变换
        return self.lin(x)


# 用户编码
class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 三层隐藏层
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        # 一层线性变换
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 门店->门店
        item_x = self.conv1(
            x_dict['store'],
            edge_index_dict[('store', 'to', 'store')],
        ).relu()
        # 门店->用户
        user_x = self.conv2(
            (x_dict['store'], x_dict['user']),
            edge_index_dict[('store', 'rev_to', 'user')],
        ).relu()
        # 中间值消息传递
        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('store', 'rev_to', 'user')],
        ).relu()

        # 一层线性变换
        return self.lin(user_x)


# 边解码
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # 二层线性变换
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        # 边标签
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        # 二层线性变换
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


# 模型
class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        # 嵌入
        self.user_emb = Embedding(num_users, hidden_channels)
        self.item_emb = Embedding(num_items, hidden_channels)
        # 编码
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        # 解码
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        # 嵌入
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['store'] = self.item_emb(x_dict['store'])
        # 编码
        z_dict['store'] = self.item_encoder(
            x_dict['store'],
            edge_index_dict[('store', 'to', 'store')],
        )
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        # 解码
        return self.decoder(z_dict['user'], z_dict['store'], edge_label_index)

import atexit
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# 运行训练
def run_train(rank, data, train_data, val_data, test_data, args, world_size):
    if rank == 0:
        print("Setting up Data Loaders...")
    train_edge_label_idx = train_data[('user', 'to', 'store')].edge_label_index
    train_edge_label_idx = train_edge_label_idx.split(
        train_edge_label_idx.size(1) // world_size, dim=1)[rank].clone()
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[8, 4],
        edge_label_index=(('user', 'to', 'store'), train_edge_label_idx),
        neg_sampling=NegativeSampling(NegativeSamplingMode.binary),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    sampled_train_data = next(iter(train_loader))
    print(sampled_train_data)

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[8, 4],
        edge_label_index=(
            ('user', 'to', 'store'),
            val_data[('user', 'to', 'store')].edge_label_index,
        ),
        edge_label=val_data[('user', 'to', 'store')].edge_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sampled_val_data = next(iter(val_loader))
    print(sampled_val_data)

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[8, 4],
        edge_label_index=(
            ('user', 'to', 'store'),
            test_data[('user', 'to', 'store')].edge_label_index,
        ),
        edge_label=test_data[('user', 'to', 'store')].edge_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sampled_test_data = next(iter(test_loader))
    print(sampled_test_data)

    def train():
        model.train()

        total_loss = total_examples = 0
        for batch in tqdm.tqdm(train_loader, disable=rank != 0):
            batch = batch.to(rank)
            optimizer.zero_grad()

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'store'].edge_label_index,
            )
            loss = F.binary_cross_entropy_with_logits(
                pred, batch['user', 'store'].edge_label)

            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_examples += pred.numel()

        return total_loss / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, targets = [], []
        for batch in tqdm.tqdm(loader, disable=rank != 0):
            batch = batch.to(rank)

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'store'].edge_label_index,
            ).sigmoid().view(-1).cpu()
            target = batch['user', 'store'].edge_label.long().cpu()

            preds.append(pred)
            targets.append(target)

        pred = torch.cat(preds, dim=0).numpy()
        target = torch.cat(targets, dim=0).numpy()

        return roc_auc_score(target, pred)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    # release all resources
    atexit.register(cleanup)
    # 模型
    model = Model(
        num_users=data['user'].num_nodes,
        num_items=data['store'].num_nodes,
        hidden_channels=64,
        out_channels=64,
    ).to(rank)
    # 输出网络结构
    print(model)
    # Initialize lazy modules
    for batch in train_loader:
        batch = batch.to(rank)
        _ = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'store'].edge_label_index,
        )
        break
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = 0
    best_val_auc = 0
    for epoch in range(1, args.epochs):
        print("Train")
        loss = train()
        if rank == 0:
            print("Val")
            val_auc = test(val_loader)
            best_val_auc = max(best_val_auc, val_auc)
        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val AUC: {val_auc:.4f}')
    if rank == 0:
        print("Test")
        test_auc = test(test_loader)
        print(f'Total {args.epochs:02d} epochs: Final Loss: {loss:4f}, '
              f'Best Val AUC: {best_val_auc:.4f}, '
              f'Test AUC: {test_auc:.4f}')
    # release all resources
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers per dataloader")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument(
        '--dataset_root_dir', type=str,
        default=osp.join(osp.dirname(osp.realpath(__file__)),
                         '../../data/LeKe'))
    args = parser.parse_args()

    def pre_transform(data):
        # Compute sparsified store<>store relationships through users:
        print('Computing store<>store relationships...')
        mat = to_scipy_sparse_matrix(data['user', 'store'].edge_index).tocsr()
        mat = mat[:data['user'].num_nodes, :data['store'].num_nodes]
        comat = mat.T @ mat
        comat.setdiag(0)
        comat = comat >= 3.
        comat = comat.tocoo()
        row = torch.from_numpy(comat.row).to(torch.long)
        col = torch.from_numpy(comat.col).to(torch.long)
        data['store', 'store'].edge_index = torch.stack([row, col], dim=0)
        return data

    # 数据集
    dataset = LeKe(args.dataset_root_dir, pre_transform=pre_transform)
    data = dataset[0]
    print(dataset)
    print(data)

    # 用户、门店
    data['user'].x = torch.arange(0, data['user'].num_nodes)
    data['store'].x = torch.arange(0, data['store'].num_nodes)

    # Only consider user<>store relationships for simplicity:
    # 为了简单起见，仅考虑用户<>门店的关系
    del data['city']
    del data['store', 'city']
    del data['user', 'store'].time
    del data['user', 'store'].behavior

    # Add a reverse ('store', 'rev_to', 'user') relation for message passing:
    # 为消息传递添加反向（'store'，'rev_to'，'user'）关系：
    data = T.ToUndirected()(data)

    # Perform a link-level split into training, validation, and test edges:
    print('Computing data splits...')
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=[('user', 'to', 'store')],
        rev_edge_types=[('store', 'rev_to', 'user')],
    )(data)
    print('Done!')

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run_train,
             args=(data, train_data, val_data, test_data, args, world_size),
             nprocs=world_size, join=True)
