import torch
from torch import nn
import os

from lib import model, trainer, dataset

print('Beginning')

parser = trainer.make_parser()
args = parser.parse_args()

print('args: ', args)

# Run Settings
save_loc = '/data/bball/save_loc'
name = 'tst'
log_dir = '/data/bball/logs'
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
seed = 1
batch_size = 32
epoch_limit = 20
criterion = nn.functional.smooth_l1_loss
gpu_device = 1
lr = 0.005
clip = 1
on_policy = False
double_q = False
num_workers = 1
reset_rate = 10000
validate_rate = 50
gamma = 1

torch.cuda.set_device(args.device)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

data = dataset.DummyBballDataset(data_size=1000, seed=seed)

model = model.NothingButNetDQN()


tr = trainer.DeepQTrainer(data=data,
                          batch_size=batch_size,
                          epoch_limit=epoch_limit,
                          criterion=criterion,
                          save_loc=save_loc,
                          name=name,
                          log_dir=log_dir,
                          gpu_device=gpu_device,
                          lr=lr,
                          clip=clip,
                          on_policy=on_policy,
                          double_q=double_q,
                          num_workers=num_workers,
                          reset_rate=reset_rate,
                          validate_rate=validate_rate)
print(name)
print('training for {} epochs'.format(epoch_limit))
tr.train(model=model, gamma=gamma)
