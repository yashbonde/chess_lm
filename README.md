# Chess LM

Current neural chess engines like AlphaZero train a network based on self-play trajectories. Now that is a bit too much for me to handle alone and neither do I have the compute necessary. So it is possible to train a supervised model on millions of games and play against it.

Initially when I first started the loss kept cycling in a particular range. However scaling up the model, training on a large batch size and reducing learning rate fixes this issue and model has consistenly reducing total loss.

### Data

To prepare your own data run the script `download.py` as follows:
```
# downloads the data from the list of links
python3 download.py -d # will take 20-30 mins depending on internet speed

# parses the data and starts storing information in text files
python3 download.py -p # will take few hours

Found 472 files.
Opening file: pgns/Sicilian2d6-4Qxd4.pgn: 100%|███████████████████| 472/472 [00:23<00:00, 20.20it/s]
Found 3167846 games, loaded 3166353 games (0.9995287018371474)
```

There are few improvements to be done if you are interested:

1. Introduce `multiprocessing` in the `download.py` to speed up the process, it was not needed in my case since I ran the code overnight on my laptop.
2. Speed up downloads by using `FTP` scripts instead of `HTTP` used by `requests` package.

Or skip this entire part and download the ZIP using the following command:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tdUgOB1VOnIT6opEJBzptp_rfUSEcbUZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tdUgOB1VOnIT6opEJBzptp_rfUSEcbUZ" -O agg.zip && rm -rf /tmp/cookies.txt
unzip agg.zip
```

This will download a zip file and extract to different files `agg_mv.txt` and `agg_res.txt` which are the moves file and results file respectively. (Read Credits for further clarification)

~~I am currently training a model on colab that has the configuration given in [`config.json`](config.json). This takes about 13 mins to go through one epoch, I am using train/finetune concept that seemlessly trains or finetunes an existting model (super useful when using colab).~~ So it has real hard time training on Colab.

### Model

The model is Language Modelling GPT2 and uses `lm_head` as policy head and another head as the value head. Value loss is MSE loss while policy uses a categorical cross entropy loss. The neat thing about the vocabulary is that instead of having a complicated method like one used in AlphaZero-CNN or having multihead system we use a relatively simple approach.

* There are total 1792 possible moves on the chess board and so instead of feeding the entire board we assume that the neural network would have a model in its hidden layers.
* Like GPT2 training we do not train seperate sequences with attention masks, instead we pass a `[NEW GAME]` token and assume that the network should be able to use this as a delimiter.

### Training

The if you have unzipped in same level as this repo then training is straightforward, run the command
```
python train.py --model=<model-name>
```

For baseline model I have following configurations:
```python
{
  "n_embd": 128,
  "n_head": 8,
  "n_inner": None,
  "n_layer": 30,
  "n_positions": 60,
  "beta1": 0.9,
  "beta2": 0.95,
  "lr": 0.0001,
}
```

I use 2x1080Ti configuration with 128 GB of RAM, `batch_size=350` seems to fill just about both the GPUs.

### Training Logs

|name|n_embd|n_layer|buffer_size|batch_size|
|-|-|-|-|-|
|v0|128|30|55555|350|
|v3|128|30|99999|350|
|v5|128|30|1000000|350|
|v6|256|20|1000000|256|
|v7|128|30|10000000|350|

**Lessons:** A couple of observations:
* Larger embedding dimension in case of `v6` did not bring any considerable difference, smaller model like v3 after training for more than 2 epochs converges at the same result. Thus meaning that larger model only learns quicker but does not have any considerable advantages.
* Larger `buffer_size`improves training. I started off with the `IterableDataset` because such a large file could not be loaded on the RAM. However with larger and more powerful computer, I am planning to load the complete dataset on the memory and use a `Dataset`.

## What's the metric?

So a natural question is how are the comparisons and testing done. Well there are a few things to compare here:

### ELO Rating

How good is a particular agent? When comparing chess players [ELO](https://en.wikipedia.org/wiki/Elo_rating_system) is the mtetric used. So this will give us the comparisons not only with humans but between different players as well. For eg. we can compare between LSTM-network vs Transformer or Distilled Models vs Full models.

For this I have added a file called `tournament.py` and most of the things will be handled using a bash script.

## Intelligence

Initially I wrote this when I started this repo:

> So after all the parsing and all, I compressed the file using a standard compressor on macOS and ~1.1GB of files was compressed to ~360MB, which maens that if my model is bigger than ~1.1GB, we cannot say it has learned something just that it has stored all the information in it's weights. If however the model size is smaller than ~360MB, model would have learned to go through compressed data. So this is a unique metric to look at. Eg. GPT-2 big was ~1.75Bn Parameters trained on 40GB of uncompressed text, even if I assume ~20 GB after standard compression (would have been more than this), the model still learned a lookup table through the compressed data.

Now after reading a bit, I was on the right track. "Lossless compression is a sign of intelligence".

## Credits

The code I have written is under MIT License, code used under its own licenses. I scraped the links from [pgnmentor.com](https://www.pgnmentor.com/files.html) and the games are by the players. I am not aware of the legality here, so if you have an problem, just raise an issue. For now I have uploaded the parsed ZIP file on my GDrive.
