import torch
from segmentation_models_pytorch.utils.train import TrainEpoch
import model as md
import prepare_data as data
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 10

model = md.get_model()
model.to(device)

loss_fn = md.get_loss_fn()
metrics = md.get_accuracy_metrics()
optimizer = md.get_optimizer_fn(model,0.001)

train_set, test_set = data.get_data()
train_loader = data.get_trainloader(train_set)
valid_loader = data.get_validloader(test_set)

train_epoch = TrainEpoch(
    model,
    loss=loss_fn,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

max_score = 0.90

start_time = time.time()

model.train()

for i in range(n_epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)

    print(train_logs)

    if max_score < train_logs['accuracy']:

        max_score = train_logs['accuracy']
        modelFileName = f'./models/model_{max_score}.pth'
        torch.save(model.state_dict(), modelFileName)
        print('Model saved!')

print('Totla time: ', time.time() - start_time)