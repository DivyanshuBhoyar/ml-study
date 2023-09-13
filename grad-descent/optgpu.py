# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %%
transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# %%
trainset = datasets.MNIST(
    "./data/MNIST_data/", download=True, train=True, transform=transformer
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True)

# %%
from torch import nn, optim
import torch.nn.functional as F


class ClassifierNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ly1 = nn.Linear(784, 128)
        self.ly2 = nn.Linear(128, 64)
        self.ly3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1).cuda()  # Move input data to GPU

        x = F.relu(self.ly1(x.cuda()))  # Perform linear transformation on GPU
        x = F.relu(self.ly2(x.cuda()))  # Perform linear transformation on GPU

        return F.log_softmax(
            self.ly3(x.cuda()), dim=1
        )  # Perform linear transformation on GPU


EPOCHS = 70
criterion = nn.NLLLoss()


# %%
def train(model, optimizer, log_title=""):
    j_history = []
    for e in range(EPOCHS):
        epoch_loss = 0
        for imgs, labels in trainloader:
            probab = model(imgs.cuda())  # Move input data to GPU
            loss = criterion(probab, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if log_title:
            print(f"{log_title} @ epoch {e+1} :: loss = {epoch_loss/len(trainloader)}")
        j_history.append(epoch_loss / len(trainloader))

    return j_history


# %% [markdown]
# #### SGD

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU

optimizer = optim.SGD(model.parameters(), lr=0.03)
sgd_losses = train(model, optimizer, "SGD")

# %% [markdown]
# ##### Momentum

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
mtm_losses = train(model, optimizer, "Momentum")

# %% [markdown]
# ##### Nesterov

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, nesterov=True)
nestv_losses = train(model, optimizer, "Nesterov")

# %% [markdown]
# ##### Adagrad

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU
optimizer = optim.Adagrad(model.parameters(), lr=0.03, eps=1e-8)
adgd_losses = train(model, optimizer, "Adagrad")

# %% [markdown]
# ##### RMSProp

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU
optimizer = optim.RMSprop(model.parameters(), lr=0.03, momentum=0.9, eps=1e-8)
rms_losses = train(model, optimizer, "RMSProp")

# %% [markdown]
# ##### Adam

# %%
model = ClassifierNN().to("cuda")  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=0.03, betas=(0.9, 0.998))
adam_losses = train(model, optimizer, "Adam")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(sgd_losses, label="SGD", color="r")
plt.plot(mtm_losses, label="Momentum", color="g")
plt.plot(nestv_losses, label="Nesterov", color="b")
plt.plot(adgd_losses, label="Adagrad", color="y")
plt.plot(rms_losses, label="RMSProp", color="m")
plt.plot(adam_losses, label="Adam", color="k")

plt.title("Loss trends")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.show()
