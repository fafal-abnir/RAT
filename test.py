import torch.nn
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from src.models.MLP import MLP
from src.models.GCN import GCN
from src.utils.visualization.visualization import visualize_embedding

dataset = Planetoid(root='datasets/Planetoid', name="Cora", transform=NormalizeFeatures())
print()
print(f'Dataset:{dataset}')
print(f"Number of graphs:{len(dataset)}")
print(f"Number of features:{dataset.num_features}")
print(f"Number of classes:{dataset.num_classes}")
data = dataset[0]

print()
print(data)
print("=" * 10)
# Information about the first graph
print(f"Number of nodes:{data.num_nodes}")
print(f"Number of edges:{data.num_edges}")
print(f"Number of training nodes:{data.train_mask.sum()}")
print(f"Number of test nodes:{data.test_mask.sum()}")
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

#### MLP testing
model = MLP(input_channels=dataset.num_features, output_channels=dataset.num_classes, hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(model)


### Training Testing


### GCN Testing
model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.eval()
embedding = model(data.x, data.edge_index)
visualize_embedding(embedding, color=data.y)


### Training/Testing GCN

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch:{epoch:03d} Loss:={loss}")

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

model.eval()

out = model(data.x, data.edge_index)
visualize_embedding(out, color=data.y)