from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aligner import GraphemeAligner
from collate import LJSpeechCollator
from dataset import LJSpeechDataset
from featurizer import MelSpectrogram, MelSpectrogramConfig
from model import FastSpeech
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fast_speech = FastSpeech().to(device)

dataloader = DataLoader(LJSpeechDataset('./data/dataset/LJSpeech'), batch_size=3, collate_fn=LJSpeechCollator())
batch = list(islice(dataloader, 1))[0]
aligner = GraphemeAligner().to(device)
batch.durations = aligner(
    batch.waveform.to(device), batch.waveform_length, batch.transcript
)
featurizer = MelSpectrogram(MelSpectrogramConfig())
waveform = batch.waveform
mels = featurizer(waveform).to(device)
tokens = batch.tokens
durations = batch.durations * mels.shape[-1]
mels = mels.transpose(1, 2)
n_epochs = 2000
output_step = 100

fast_speech.train()
preds_loss = nn.MSELoss()
durations_loss = nn.MSELoss()
print("Params: ", sum(p.numel() for p in fast_speech.parameters() if p.requires_grad))
optimizer = optim.Adam(fast_speech.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

for epoch in range(1, n_epochs + 1):
    preds, duration_preds = fast_speech(tokens, durations)
    optimizer.zero_grad()
    common_shape = min(preds.size(1), mels.size(1))
    loss = preds_loss(preds[:, :common_shape, :], mels[:, :common_shape, :]) + durations_loss(torch.exp(duration_preds), durations)
    loss.backward()
    optimizer.step()
    if epoch % output_step == 0:
        print(f"Epoch #{epoch} loss: {loss.item()}")
