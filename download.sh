mkdir -p data/dataset/LJSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 --output-document data/dataset/LJSpeech/LJSpeech-1.1.tar.bz2
tar -xjf data/dataset/LJSpeech/LJSpeech-1.1.tar.bz2 --directory=data/dataset/LJSpeech
rm data/dataset/LJSpeech/LJSpeech-1.1.tar.bz2
