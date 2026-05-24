import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
     super().__init__()

     self.Layer1 = nn.Linear(3,4)
     self.Layer2 = nn.Linear(4,1)

     self.relu = nn.ReLU()
     self.sigmoid = nn.Sigmoid()

    def forward(self,x):
       x = self.Layer1(x)
       x = self.relu(x)
       x = self.Layer2(x)
       x = self.sigmoid(x)
       return x
    

model = NeuralNetwork()

x =torch.randn(5,3)
y =torch.tensor([[1.0],[0.0],[1.0],[0.0],[1.0]])

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1000):
   prediction = model(x)
   loss = loss_fn(prediction,y)

   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   if epoch % 10 ==0:
      print(f"Epochs {epoch}, Loss:{loss.item()}")




with torch.no_grad():
    output = model(x)
    predicted = (output > 0.5).float()  # threshold at 0.5
    print(predicted)