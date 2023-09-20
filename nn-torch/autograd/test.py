import torch
import numpy as np
from tensor import MyTensor

# Your MyTensor class goes here


def compare_with_pytorch():
    data1 = np.random.rand(3, 3)
    data2 = np.random.rand(3, 3)

    my_tensor1 = MyTensor(data1)
    my_tensor2 = MyTensor(data2)

    torch_tensor1 = torch.tensor(data1, requires_grad=True)
    torch_tensor2 = torch.tensor(data2, requires_grad=True)

    my_result = my_tensor1 + my_tensor2
    my_result = my_result * my_tensor1
    my_result = my_result**2
    my_result = my_result.exp()
    my_result = my_result.tanh()
    my_result = my_result.elu()
    my_result = my_result.softplus()
    my_result._backward()

    elufn = torch.nn.ELU()

    torch_result = torch_tensor1 + torch_tensor2
    torch_result = torch_result * torch_tensor1
    torch_result = torch_result**2
    torch_result = torch_result.exp()
    torch_result = torch_result.tanh()
    torch_result = elufn(torch_result)  # Corrected line
    torch_result = torch.nn.functional.softplus(torch_result)
    torch_result.backward(torch.ones_like(torch_result))

    print("MyTensor result:", my_result.data)
    print("PyTorch result:", torch_result.data.numpy())


compare_with_pytorch()
