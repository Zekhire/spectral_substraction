# Spectral substraction


#### Technologies
Project was created and tested with:
* Windows 10
* Python 3.6.5


#### Description
Project used to denoise noisy audio files. This project uses spectral substraction method (both general and not general version). Currently only .wav files are supported.


#### Setup
- Run following block of commands in spectral_substraction\ catalogue:
```
python -m virtualenv venv
cd venv
cd Scripts
activate
cd ..
cd ..
pip install -r requirements.txt
```
- Set all paths for audio files in "START OF THE SCRIPT" section
- Set all parameters in "START OF THE SCRIPT" section


#### Run
Go to spectral_substraction\ and run command:
```
python spectral_substraction.py
```


#### References
This project is created based on following document (pages 82-85):
https://eti.pg.edu.pl/documents/176593/26756916/STS.pdf