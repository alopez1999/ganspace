import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import time
import math

batch_size = 128
epochs = 30
learning_rate = 1e-3
resume_training = True
compute_test_loss = False
save_ckpt = False
normalize = True
PATH = "autoencoder/checkpoint_norm.pt"

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_in = nn.Linear(
            in_features=512, out_features=480
        )
        self.encoder_1 = nn.Linear(
            in_features=480, out_features=400
        )
        self.encoder_2 = nn.Linear(
            in_features=400, out_features=300
        )
        self.encoder_3 = nn.Linear(
            in_features=300, out_features=200
        )
        self.encoder_4 = nn.Linear(
            in_features=200, out_features=100
        )
        self.encoder_code = nn.Linear(
            in_features=100, out_features=69
        )

        self.decoder_code = nn.Linear(
            in_features=69, out_features=100
        )
        self.decoder_1 = nn.Linear(
            in_features=100, out_features=200
        )
        self.decoder_2 = nn.Linear(
            in_features=200, out_features=300
        )
        self.decoder_3 = nn.Linear(
            in_features=300, out_features=400
        )
        self.decoder_4 = nn.Linear(
            in_features=400, out_features=480
        )
        self.decoder_out = nn.Linear(
            in_features=480, out_features=512
        )


    def encoder_pass(self, features):
        activation = self.encoder_in(features)
        activation = torch.relu(activation)
        activation = self.encoder_1(activation)
        activation = torch.relu(activation)
        activation = self.encoder_2(activation)
        activation = torch.relu(activation)
        activation = self.encoder_3(activation)
        activation = torch.relu(activation)
        activation = self.encoder_4(activation)
        activation = torch.relu(activation)
        code = self.encoder_code(activation)
        code = torch.relu(code)
        return code

    def decoder_pass(self, code):
        activation = self.decoder_code(code)
        activation = torch.relu(activation)
        activation = self.decoder_1(activation)
        activation = torch.relu(activation)
        activation = self.decoder_2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_3(activation)
        activation = torch.relu(activation)
        activation = self.decoder_4(activation)
        activation = torch.relu(activation)
        activation = self.decoder_out(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    def forward(self, features):
        code = self.encoder_pass(features)
        reconstructed = self.decoder_pass(code)
        return reconstructed

class LatentDataset(Dataset):
    def __init__(self, data):
        self.x = data
        self.n_samples = data.shape[0] 
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return self.n_samples

data = np.load("cache/components/LatentSamples.npy")
#data=data[:100000,:]

if normalize:
    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, keepdims=True)
    data = (data - means) / stds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
print("Model Generated!")


n_samples = data.shape[0] 
trainData=torch.Tensor(data[:n_samples-batch_size*2,:])
validationData = torch.Tensor(data[n_samples-batch_size*2:n_samples-batch_size,:])
testData = torch.Tensor(data[n_samples-batch_size:,:])

train_dataset = LatentDataset(trainData)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print ("Data loaded!")


if resume_training:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_ckpt = checkpoint['epoch']
    loss_ckpt = checkpoint['loss']
    model.train()
    print("Resuming training from epoch : {} with train_loss {:.6f}".format(epoch_ckpt, loss_ckpt))
else:
    epoch_ckpt = 0

start = time.time()
for epoch in range(epochs):
    loss = 0
    epoch_start = time.time()
    for i, batch in enumerate(train_dataloader):
        batch_features = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    
    loss = loss / len(train_dataloader)

    # compute validation loss
    with torch.no_grad():
        inputs = validationData.to(device)
        outputs = model(inputs)
        val_loss = criterion(outputs, inputs)

    epoch_end = time.time()
    epoch_time = epoch_end-epoch_start
    # display the epoch training loss
    print("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}, time = {:02.0f}s".format(epoch + 1 + epoch_ckpt, epochs + epoch_ckpt, loss, val_loss, epoch_time))

    if (((epoch % 10) == 0) & save_ckpt):
        torch.save({
                'epoch': epoch_ckpt + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'time': "--"
                }, PATH)
        print ("Model saved to checkpoint in epoch {}".format(epoch_ckpt + epoch + 1))

#time estimation
end = time.time()
time_elapsed = (end-start)
hours = math.floor(time_elapsed/3600)
minutes = math.floor((time_elapsed % 3600)/60)
seconds = ((time_elapsed % 3600) % 60) % 60
print("training time {:02.0f}:{:02.0f}:{:02.0f} (hh:mm:ss)".format(hours, minutes, seconds))

# compute test loss
if compute_test_loss:
    with torch.no_grad():
        inputs = testData.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, inputs)
    print("Loss on test dataset: {:.6f}".format(test_loss))

if save_ckpt:
    torch.save({
                'epoch': epoch_ckpt + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'time': time_elapsed
                }, PATH)
    print("Job succesfully finished and saved to checkpoint!")