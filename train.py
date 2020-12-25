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
args.add_argument("--maxlen", type = int, default = 85 * 2, help = "maximum length")
args.add_argument("--ds", type = str, default = "full", help = "to use in memory or iterable Dataset [full / iter]")
args.add_argument("--buffer", type = int, default = 99999, help = "buffer size for DataSet")
args.add_argument("--n_embd", type = int, default = 200, help = "embedding dim")
args.add_argument("--n_layer", type = int, default = 10, help = "number of layers of the model")
args.add_argument("--n_head", type = int, default = 10, help = "number of heads for MHA")
args.add_argument("--model", type = str, default = "beta", help = "which model to train, select from `base`, `beta`")
args.add_argument("--use_drop", type = bool, default = False, help = "set `True` to use dropout wuith value 0.1")

# optim settings
args.add_argument("--lr", type = float, default = 1e-4, help = "learning rate")
args.add_argument("--beta1", type = int, default = 0.9, help = "Adam.beta1") # momentum for first gradient
args.add_argument("--beta2", type = int, default = 0.95, help = "Adam.beta2") # momentum for second moment (var)

# train args
args.add_argument("--scheduler", type=str, default = "GPT3", help= "LR scheduler one of `CosineAnnealingWarmRestarts,"
    "OneCycleLR, MultiStepLR, NoamDecay, CosineDecay, WarmupConstant, GPT3`"
)
args.add_argument("--batch_size", type=int, default=150, help="batch size")
args.add_argument("--split", type=float, default=0.01, help="ratio of data to use as testing")
args.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train / finetune")
args.add_argument("--warmup_perc", type=float, default=0.2, help="weight decay value for L2 reg.")
args.add_argument("--weight_decay", type=float, default=0.1, help="weight decay value for L2 reg.")
args.add_argument("--save_folder", type=str, default="models", help="Folder to save model to")
args.add_argument("--name", type=str, default="cgpt", help="Saved name to have filepath `<name>.pt`")
args.add_argument("--test_every", type=int, default=1000, help="Test after these global steps")
args.add_argument("--patience", type=int, default=1, help="Early stopping partience value")
args = args.parse_args()

# path and file management
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_folder = os.path.join(args.save_folder, args.name)
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_path = os.path.join(model_folder, args.name + ".pt")

set_seed(4)

if args.lmtrain[-4:] == "hdf5":
    print("Using HDF5 data")
    assert args.maxlen % 85 == 0, "using hdf5 means maxlen % 85 == 0"
    
elif args.lmtrain[-3:] == "npz":
    print("Using numpy zips")
    assert args.maxlen % 85 == 0, "using numpy means maxlen % 85 == 0"

dataConfig = DataConfig(
    lm=args.lmtrain,
    rf=args.res,
    m2id=args.m2id,
    maxlen=args.maxlen,
    buffer= args.buffer,
)
dstrain, dstest = get_datasets(dataConfig, args.split)
print(dataConfig)

# use dropout if defined
drop_val = 0.1 if args.use_drop else 0.0
modelConfig = ModelConfig(
    vocab_size = len(dstrain.m2id),
    n_positions=args.maxlen,
    n_ctx=args.maxlen,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    loss_method = "mse", # add loss method for values
    resid_pdrop=drop_val,
    embd_pdrop=drop_val,
    attn_pdrop=drop_val,
)
print(modelConfig)
if args.model == "beta":
    model = BetaChess(modelConfig)
elif args.model == "base":
    model = BaseHFGPT(modelConfig)
elif args.model == "value":
    model = ValueOnlyNetwork(modelConfig)
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
    weight_decay = args.weight_decay,
    
    warmup_perc = args.warmup_perc, # 14% of steps are used for warmup
    jitter_scale = 20, # scale for LR noise
    
    # get 19305 batches with batch size = 185 with 170 tokens
    # final_tokens = total_batches * batch_size * toks in each batch size
#     final_tokens = 19305 * 185 * 170, # value at which the lr goes to 10% of original
#     warmup_tokens = 19305 * 185 * 170, # 10 % of the steps are warmup steps
    
    # n_layer = 10 and bactch_size = 150 gives 23809 steps
#     final_tokens = 23809 * 150 * 170,
)
trainer = Trainer(model, dstrain, trainerConf, dstest)
print(trainerConf)
trainer.train(args)
