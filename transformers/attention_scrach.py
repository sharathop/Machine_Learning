import torch
import torch.nn.functional as F



def attention_score(W, K, V, mask=None):

    score = torch.matmul(W, K.transpose(-2, -1))

    d_k = K.size(-1)
    score = score / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        score = score.masked_fill(mask==0, -1e9)

    
    attention_weight = F.softmax(score, dim =-1)

    output =torch.matmul(attention_weight, V)

    return output, attention_weight



W = torch.tensor([[
    [1.,0.,1.],
    [0.,1.,0.],
    [1.,0.,0]
]])

K = torch.tensor([[
    [1.,1.,1.],
    [0.,1.,0.],
    [1.,1.,0]
]])

V = torch.tensor([[
    [0.,0.,1.],
    [0.,1.,0.],
    [1.,1.,0.]
]])


output ,weight = attention_score(W,K,V)


print("Attention Weights:")
print(weight)

print("\noutput:")
print(output)

