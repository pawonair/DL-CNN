@echo off
curl.exe -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
if exist cifar-10-python.tar.gz del /F /Q cifar-10-python.tar.gz
