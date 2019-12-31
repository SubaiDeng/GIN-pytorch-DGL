import dgl
import torch
a = list([1, 2, 1])
b = list([3, 2, 3])
c = torch.tensor([3, 3, 2])
g = dgl.DGLGraph()
g.add_nodes(5)
g.add_edges(a, b)
g.edata['w'] = c
print(g.edges[0].data)
print("Finished")
