# Chess LM

A transformer fit to chess data to see if it performs better than most of players. To get the data run the scrip `download.py` as follows:
```
# downloads the data from the list of links
python3 download.py -d # will take 20-30 mins depending on internet speed

# parses the data and starts storing information in text files
python3 download.py -p # will take few hours

# legacy
python3 download.py -v # prints the meta

Found 472 files.
Opening file: pgns/Sicilian2d6-4Qxd4.pgn: 100%|███████████████████| 472/472 [00:23<00:00, 20.20it/s]
Found 3167846 games, loaded 3166353 games (0.9995287018371474)
```

There are few improvements to be done if you are interested:

1. Introduce `multiprocessing` in the `download.py` to speed up the process, it was not needed in my case since I ran the code overnight on my laptop.
2. Speed up downloads by using `FTP` scripts instead of `HTTP` used by `requests` package.


I am currently training a model on colab that has the configuration given in [`config.json`](config.json). This takes about 13 mins to go through one epoch, I am using train/finetune concept that seemlessly trains or finetunes an existting model (super useful when using colab).

## What's the metric?

So a natural question is how are the comparisons and testing done. Well there are a few things to compare here:

### ELO Rating

How good is a particular agent? When comparing chess players [ELO](https://en.wikipedia.org/wiki/Elo_rating_system) is the mtetric used. So this will give us the comparisons not only with humans but between different players as well. For eg. we can compare between LSTM-network vs Transformer or Distilled Models vs Full models.

### Model

So after all the parsing and all, I compressed the file using a standard compressor on macOS and ~1.1GB of files was compressed to ~360MB, which maens that if my model is bigger than ~1.1GB, we cannot say it has learned something just that it has stored all the information in it's weights. If however the model size is smaller than ~360MB, model would have learned to go through compressed data. So this is a unique metric to look at.

Eg. GPT-2 big was ~1.75Bn Parameters trained on 40GB of uncompressed text, even if I assume ~20 GB after standard compression (would have been more than this), the model still learned a lookup table through the compressed data.

**Update 09.08.2020:** The model I am training on Colab has the following configurations (`211040 * 32 bits = 844160 Bytes` or `0.85KB` model which is `<< 360MB`):
```
2020-08-09 17:30:30.061572: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
GPT2Config {
  "activation_function": "gelu_new",
  "attn_pdrop": 0.0,
  "batch_size": 4512,
  "bos_token_id": 50256,
  "embd_pdrop": 0.0,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "maxlen": 30,
  "model_type": "gpt2",
  "n_ctx": 30,
  "n_embd": 32,
  "n_head": 2,
  "n_layer": 12,
  "n_positions": 30,
  "optimizer": "Adam",
  "optimizer_params": {
    "betas": [
      0.9,
      0.999
    ],
    "lr": 0.003
  },
  "resid_pdrop": 0.0,
  "summary_activation": null,
  "summary_first_dropout": 0.0,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "vocab_size": 1799
}

Model Size: 211040
```

## Credits

I scraped the links from [pgnmentor.com](https://www.pgnmentor.com/files.html) and the games are by the players. I am not aware of the legality here, so if you have an problem, just raise an issue with the appropriate tag.