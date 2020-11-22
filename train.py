"""Finetuning GPT2 to play chess
@yashbonde-09.08.2020"""

import os
from argparse import ArgumentParser
from model import DataConfig, ChessData, ChessDataInMemory, ModelConfig, BaseHFGPT, TrainerConfig, Trainer

# load user args
args = ArgumentParser(description="Train GPT2 model on t2sql corpus")

# data args
args.add_argument("--lmtrain", type=str, default = "agg_mv.txt", help="path to train_lm file")
args.add_argument("--lmtest", type=str, default = None, help="path to test_lm file")
args.add_argument("--res", type=str, default = "agg_res.txt", help="path to res file")
args.add_argument("--m2id", type=str, default = "moves.json", help="path to move_to_id json")

args.add_argument("--maxlen", type = int, default = 60, help = "maximum length")

args.add_argument("--ds", type = str, default = "full", help = "to use in memory or iterable Dataset [full / iter]")
args.add_argument("--buffer", type = int, default = 99999, help = "buffer size for DataSet")
args.add_argument("--n_embd", type = int, default = 128, help = "embedding dim")
args.add_argument("--n_layer", type = int, default = 30, help = "number of layers of the model")
args.add_argument("--n_head", type = int, default = 8, help = "number of heads for MHA")

args.add_argument("--lr", type = int, default = 0.0001, help = "learning rate")
args.add_argument("--beta1", type = int, default = 0.9, help = "Adam.beta1")
args.add_argument("--beta2", type = int, default = 0.95, help = "Adam.beta2")

# train args
args.add_argument("--batch_size", type=int, default=350, help="batch size")
args.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train / finetune")
args.add_argument("--save_folder", type=str, default="models", help="Folder to save model to")
args.add_argument("--model", type=str, default="cgpt", help="Saved model to have filepath `<model>.pt`")
args.add_argument("--save_every", type=int, default=1000, help="save this global steps")
args = args.parse_args()

# path and file management
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_folder = os.path.join(args.save_folder, args.model)
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_path = os.path.join(model_folder, args.model + ".pt")

dataConfig = DataConfig(
    lm=args.lmtrain,
    rf=args.res,
    m2id=args.m2id,
    maxlen=args.maxlen,
    buffer= args.buffer,
)
if args.ds == "full":
    dstrain = ChessDataInMemory(dataConfig)
else:
    dstrain = ChessData(dataConfig)

modelConfig = ModelConfig(
    vocab_size = len(dstrain.m2id),
    n_positions=args.maxlen,
    n_ctx=args.maxlen,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head
)
model = BaseHFGPT(modelConfig)
print(f"Model Size: {sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")

trainerConf = TrainerConfig(
    max_epochs = args.num_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    betas = (args.beta1, args.beta2),
    tb_path = model_folder,
    save_every = args.save_every,
    ckpt_path = model_path
)
trainer = Trainer(model, dstrain, trainerConf)
trainer.train()
