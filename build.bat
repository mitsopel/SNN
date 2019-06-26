@echo off

"C:\Program Files (x86)\Python36-32\python" Source/main.py  --normalize 1 --batch-size 100 --activation 4 --hidden-count 100 --learning-rate 1.0 --lambda-normalization 0.0001 --grad-check 0 --epochs 10 --test-dir "Resources\Test" --train-dir "Resources\Train" --feature-count -1 --test-count -1
