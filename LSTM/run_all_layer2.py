import os

# 2-layer lstm, seeds: 0, 1, 2
for seed in [0, 1, 2]:
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save LSTM-PTB --when 100 145 --clip 0.15 '
              f'--beta1 0.9 --beta2 0.999 --optimizer sgdf --lr 60 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.001 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.01 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer msvag --lr 30 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.001 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer fromage --lr 0.01 --eps 1e-8 --nlayer 2 --seed {seed}')
    
    os.system(f'python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 '
              f'--epoch 200 --save PTB.pt --when 100 145 --clip 0.25 '
              f'--beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.001 --eps 1e-8 --nlayer 2 --seed {seed}')
