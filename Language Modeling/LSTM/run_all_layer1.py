import os

# 1-layer lstm, seeds: 0, 1, 2
for seed in [0, 1, 2]:
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save LSTM-PTB --when 100 145 --clip 0.1 --beta1 0.9 --beta2 0.999 '
              f'--optimizer sgdf --lr 60 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer sgd --lr 30 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer adam --lr 0.001 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer radam --lr 0.001 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer msvag --lr 30 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer adabound --lr 0.001 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer fromage --lr 0.01 --eps 1e-8 --nlayer 1 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save LSTM-PTB --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 '
              f'--optimizer adabelief --lr 0.001 --eps 1e-8 --nlayer 1 --seed {seed}')
