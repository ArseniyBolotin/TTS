# TTS

FastSpeech implementation for Text-to-Speech task

## Train reproduce

```shell
git clone https://github.com/ArseniyBolotin/TTS.git
pip3 install -r TTS/requirements.txt
cd ./TTS && sh ./download.sh
cd ./TTS && python3 train.py
```

### Clone repo
```shell
git clone https://github.com/ArseniyBolotin/TTS.git
```


### Install requirements
```shell
pip3 install -r TTS/requirements.txt
```


### Download necessary files
```shell
cd ./TTS && sh ./download.sh
```

### Train
```shell
cd ./TTS && python3 train.py
```

### Continue training from checkpoint
Put weights in ./resume.pt
