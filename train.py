"""Finetuning GPT2 to play chess
@yashbonde-09.08.2020"""

import os
from argparse import ArgumentParser
from model import *

# load user args
args = ArgumentParser(description="Train a GPT2 model to play chess on moves only")

# data args
args.add_argument("--lmtrain", type=str, default = "data/clm.npz", help="path to train_lm file")
args.add_argument("--res", type=str, default = "data/all_res.txt", help="path to res file")
args.add_argument("--m2id", type=str, default = "assets/moves.json", help="path to move_to_id json")

# model things
args.add_argument("--maxlen", type = int, default = 85, help = "maximum length")
args.add_argument("--ds", type = str, default = "full", help = "to use in memory or iterable Dataset [full / iter]")
args.add_argument("--buffer", type = int, default = 99999, help = "buffer size for DataSet")
args.add_argument("--n_embd", type = int, default = 128, help = "embedding dim")
args.add_argument("--n_layer", type = int, default = 30, help = "number of layers of the model")
args.add_argument("--n_head", type = int, default = 8, help = "number of heads for MHA")
args.add_argument("--model", type = str, default = "beta", help = "which model to train, select from `base`, `beta`")

# optim settings
args.add_argument("--lr", type = int, default = 0.05, help = "learning rate")
args.add_argument("--beta1", type = int, default = 0.9, help = "Adam.beta1")
args.add_argument("--beta2", type = int, default = 0.95, help = "Adam.beta2")

# train args
args.add_argument("--scheduler", type=str, default=None, help= "which LR scheduler to use of `CosineAnnealingWarmRestarts` or `OneCycleLR`")
args.add_argument("--batch_size", type=int, default=270, help="batch size")
args.add_argument("--split", type=float, default=0.01, help="ratio of data to use as testing")
args.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train / finetune")
args.add_argument("--save_folder", type=str, default="models", help="Folder to save model to")
args.add_argument("--name", type=str, default="cgpt", help="Saved name to have filepath `<name>.pt`")
args.add_argument("--test_every", type=int, default=500, help="Test after these global steps")
args.add_argument("--patience", type=int, default=3, help="Early stopping partience value")
args = args.parse_args()

# path and file management
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_folder = os.path.join(args.save_folder, args.name)
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_path = os.path.join(model_folder, args.name + ".pt")

if args.lmtrain[-4:] == "hdf5":
    print("Using HDF5 data")
    assert args.maxlen == 85, "using hdf5 means you have harcoded len = 85"
    
elif args.lmtrain[-3:] == "npz":
    print("Using numpy zips")
    assert args.maxlen == 85, "using numpy means you have harcoded len = 85"

dataConfig = DataConfig(
    lm=args.lmtrain,
    rf=args.res,
    m2id=args.m2id,
    maxlen=args.maxlen,
    buffer= args.buffer,
)
dstrain, dstest = get_datasets(dataConfig, args.split)

modelConfig = ModelConfig(
    vocab_size = len(dstrain.m2id),
    n_positions=args.maxlen,
    n_ctx=args.maxlen,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    activation_function="relu",
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)
if args.model == "beta":
    model = BetaChess(modelConfig)
elif args.model == "base":
    model = BaseHFGPT(modelConfig)
else:
    raise ValueError(f"Found wrong model name: {args.model}")
print(f"Model Size: {sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")

trainerConf = TrainerConfig(
    num_epochs = args.num_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    betas = (args.beta1, args.beta2),
    tb_path = model_folder,
    test_every = args.test_every,
    ckpt_path = model_path,
    patience = args.patience,
    scheduler=args.scheduler,
    t0div = 5,
    tmult = 2,
)
trainer = Trainer(model, dstrain, trainerConf, dstest)
trainer.train()
