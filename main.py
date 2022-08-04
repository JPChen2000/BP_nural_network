import numpy as np
from Dataset import Dataset,DataLoader
import matplotlib.pyplot as plt
from Sequential import Sequential
from Layer import *
from Optimizer import *
# x_path = "./x.npy"
# y_path = "./y.npy"
#
# x = np.load(x_path)
# y = np.load(y_path)


class LinearDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


num_data = 200  # 训练数据数量
val_number = 500  # 验证数据数量

epoches = 1000
batch_size = 16
learning_rate = 0.01

X = np.linspace(-np.pi, np.pi, num_data).reshape(num_data, 1)
Y = np.sin(X) * 2 + (np.random.rand(*X.shape) - 0.5) * 0.1
y_ = np.sin(X) * 2

model = Sequential(
    Linear(1, 16, name='linear1'),
    ReLU(name='relu1'),
    Linear(16, 64, name='linear2'),
    ReLU(name='relu1'),
    Linear(64, 16, name='linear2'),
    ReLU(name='relu1'),
    Linear(16, 1, name='linear2'),
)
opt = SGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=0.0, decay_type='l2')
loss_fn = MSE()

train_dataset = LinearDataset(X, Y)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, drop_last=True)
for epoch in range(1, epoches):
    for x, y in train_dataloader:
        pred = model(x)

        loss = loss_fn(pred, y)

        grad = loss_fn.backward()
        model.backward(grad)

        opt.step()
        opt.clear_grad()
    print("epoch: {}. loss: {}".format(epoch, loss))

X_val = np.linspace(-np.pi, np.pi, val_number).reshape(val_number, 1)
Y_val = np.sin(X_val) * 2
val_dataset = LinearDataset(X_val, Y_val)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=2, drop_last=False)
all_pred = []
for x, y in val_dataloader:
    pred = model(x)
    all_pred.append(pred)
all_pred = np.vstack(all_pred)

plt.scatter(X, Y, marker='x')
plt.plot(X_val, all_pred, color='red')
plt.savefig("./result.png")
plt.show()

