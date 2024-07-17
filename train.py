import torch
import model as md
import prepare_data as data

n_epochs = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = md.get_model()
model.to(device)

loss_fn = md.get_loss_fn()
optimize_fn = md.get_optimizer_fn(model, 0.001)

train_set, _ = data.get_data()
train_loader = data.get_trainloader(train_set)


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimize_fn.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimize_fn.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), './models/model.pth')


for epoch in range(1, n_epochs + 1):
    train(epoch)
    print()
