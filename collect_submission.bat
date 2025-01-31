@echo off
if exist assignment4_submission.zip del /F /Q assignment4_submission.zip
tar -a -c -f assignment4_submission.zip 01_cnn/01_scratch/modules 01_cnn/01_scratch/optimizer 01_cnn/01_scratch/trainer.py 01_cnn/01_scratch/train.py 01_cnn/01_scratch/train.png 01_cnn/02_pytorch/checkpoints 01_cnn/02_pytorch/models 01_cnn/02_pytorch/main.py 02_regularization/code/*.ipynb