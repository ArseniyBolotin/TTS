from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aligner import GraphemeAligner
from collate import LJSpeechCollator
from dataset import LJSpeechDataset
from featurizer import MelSpectrogram, MelSpectrogramConfig
from model import FastSpeech
from vocoder import Vocoder
from wandb_writer import WandbWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Google drive saving
# --------------------------------------------------------------
google_drive = False
if google_drive:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.discovery import build
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
# --------------------------------------------------------------


fast_speech = FastSpeech().to(device)
dataloader = DataLoader(LJSpeechDataset('./data/dataset/LJSpeech'), batch_size=3, collate_fn=LJSpeechCollator())
aligner = GraphemeAligner().to(device)
wandb_writer = WandbWriter()
vocoder = Vocoder().to(device).eval()
featurizer = MelSpectrogram(MelSpectrogramConfig())

n_iters = 20000
save_step = 1000
output_step = 10

fast_speech.train()
criterion = nn.MSELoss()
durations_criterion = nn.MSELoss()

optimizer = optim.Adam(fast_speech.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

current_iter = 1

for batch in cycle(dataloader):
    waveform = batch.waveform
    mels = featurizer(waveform).to(device)
    tokens = batch.tokens.to(device)
    batch.durations = aligner(
        batch.waveform.to(device), batch.waveform_length, batch.transcript
    )[:, :tokens.shape[-1]]
    durations = batch.durations.to(device) * tokens.shape[-1]
    mels = mels.transpose(1, 2)
    preds, duration_preds = fast_speech(tokens, durations)
    optimizer.zero_grad()
    common_shape = min(preds.size(1), mels.size(1))
    durations_loss = durations_criterion(duration_preds, durations)
    prediction_loss = criterion(preds[:, :common_shape, :], mels[:, :common_shape, :])
    loss = prediction_loss + durations_loss
    loss.backward()
    optimizer.step()
    if current_iter % output_step == 0:
        wandb_writer.set_step(current_iter)
        wandb_writer.add_scalar("Total loss", loss.item())
        wandb_writer.add_scalar("Durations predictions loss", durations_loss.item())
        wandb_writer.add_scalar("Predictions loss", prediction_loss.item())
        wandb_writer.add_text("Text sample", batch.transcript[0])
        wandb_writer.add_audio("Ground truth audio", batch.waveform[0], sample_rate=22050)
        try:
            fast_speech.eval()
            reconstructed_wav = vocoder.inference(fast_speech(tokens, durations)[0][:1].transpose(1, 2)).cpu()
            wandb_writer.add_audio("Reconstructed audio", reconstructed_wav, sample_rate=22050)
        except RuntimeError:
            print("Iteration : ", current_iter)
            print("Too short duration predicts")
        finally:
            fast_speech.train()
    if current_iter % save_step == 0 and google_drive:
        print("Iteration : ", current_iter)
        name = 'fast_speech_' + str(current_iter) + '.pt'
        torch.save(fast_speech, name)
        drive_service = build('drive', 'v3')
        file_metadata = {'name': name}
        media = MediaFileUpload(name, resumable=True)
        created = drive_service.files().create(body=file_metadata,
                                               media_body=media,
                                               fields='id').execute()
        print("Save model")
        print('File ID: {}'.format(created.get('id')))

    current_iter += 1
    if current_iter > n_iters:
        break
