import torch
import torch.nn as nn

x =torch.tensor([[1,2],[3,4],[5,6],[7,8]], dtype=torch.float32)
y =torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
print(x)
print(x.shape)
print(y.shape)

model =nn.Linear(2,1)


criterion =nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr =0.001)

for epoch in range(100):
    pred =model(x)
    loss = criterion(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[9,10]], dtype=torch.float32)))

