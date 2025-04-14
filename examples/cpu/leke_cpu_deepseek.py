# An CPU implementation of unsupervised bipartite GraphSAGE
# using the LeKe dataset.
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import Embedding, Linear
from torch_geometric.sampler.base import NegativeSamplingMode

from torch_geometric.sampler import NegativeSampling

import torch_geometric.transforms as T
from examples.cpu.leke import LeKe
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict['store'],
            edge_index_dict[('store', 'to', 'store')],
        ).relu()

        user_x = self.conv2(
            (x_dict['store'], x_dict['user']),
            edge_index_dict[('store', 'rev_to', 'user')],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('store', 'rev_to', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.item_emb = Embedding(num_items, hidden_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['store'] = self.item_emb(x_dict['store'])
        z_dict['store'] = self.item_encoder(
            x_dict['store'],
            edge_index_dict[('store', 'to', 'store')],
        )
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict['user'], z_dict['store'], edge_label_index)


def run_train(data, train_data, val_data, test_data, args):
    print("Setting up Data Loaders...")
    train_edge_label_idx = train_data[('user', 'to', 'store')].edge_label_index.clone()

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[8, 4],
        edge_label_index=(('user', 'to', 'store'), train_edge_label_idx),
        neg_sampling=NegativeSampling(NegativeSamplingMode.binary),
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
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
        # num_workers=args.num_workers,
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
        # num_workers=args.num_workers,
    )
    sampled_test_data = next(iter(test_loader))
    print(sampled_test_data)

    def train():
        model.train()

        total_loss = total_examples = 0
        for batch in tqdm.tqdm(train_loader):
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
        for batch in tqdm.tqdm(loader):
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

    model = Model(
        num_users=data['user'].num_nodes,
        num_items=data['store'].num_nodes,
        hidden_channels=64,
        out_channels=64,
    )
    # 输出网络结构
    print(model)
    # Initialize lazy modules
    for batch in train_loader:
        _ = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'store'].edge_label_index,
        )
        break

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = 0
    best_val_auc = 0
    for epoch in range(1, args.epochs):
        print("Train")
        loss = train()
        print("Val")
        val_auc = test(val_loader)
        best_val_auc = max(best_val_auc, val_auc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val AUC: {val_auc:.4f}')
    print("Test")
    test_auc = test(test_loader)
    print(f'Total {args.epochs:02d} epochs: Final Loss: {loss:4f}, '
          f'Best Val AUC: {best_val_auc:.4f}, '
          f'Test AUC: {test_auc:.4f}')

root_path = osp.join(osp.dirname(osp.realpath(__file__)),
                         '../../data/LeKe')
print(root_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers per dataloader")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset_root_dir', type=str, default=root_path)
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

    dataset = LeKe(args.dataset_root_dir, pre_transform=pre_transform)
    print(dataset)
    data = dataset[0]
    print(data)

    data['user'].x = torch.arange(0, data['user'].num_nodes)
    data['store'].x = torch.arange(0, data['store'].num_nodes)

    # Only consider user<>store relationships for simplicity:
    del data['city']
    del data['store', 'city'].edge_index
    del data['user', 'store'].time
    del data['user', 'store'].behavior

    # Add a reverse ('store', 'rev_to', 'user') relation for message passing:
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

    print('Using CPU...')
    run_train(data, train_data, val_data, test_data, args)
