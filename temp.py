from torch.nn.utils import prune
import torch


m = torch.nn.Linear(1,1)
w_i = m.weight.data.detach().clone()
b_i = m.bias.data.detach().clone()
data = torch.randn(20,2)
x = data[:,0].view(-1,1)
y = 4*x+2

epochs = 30
optim = torch.optim.Adam(m.parameters())

print(m(x).shape)

for i in range(epochs):
    optim.zero_grad()
    pred = m(x)
    loss = ((pred-y)**2).sum()
    loss.backward()
    optim.step()

print("w_i:",w_i)
print("b_i:",b_i)
print("\n\nw_f:",m.weight.data)
print("b_f:",m.bias.data)



class CustomPruningSon(prune.BasePruningMethod):
    PRUNING_TYPE='unstructured'
    def compute_mask(self, t, default_mask):
        return default_mask

def customprune_unstructured(module, name):
    CustomPruningSon.apply(module, name)
    return module
