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

## What's the metric?

So a natural question is how are the comparisons and testing done. Well there are a few things to compare here:

### ELO Rating

How good is a particular agent? When comparing chess players [ELO](https://en.wikipedia.org/wiki/Elo_rating_system) is the mtetric used. So this will give us the comparisons not only with humans but between different players as well. For eg. we can compare between LSTM-network vs Transformer or Distilled Models vs Full models.

### Model

So after all the parsing and all, I compressed the file using a standard compressor on macOS and ~1.1GB of files was compressed to ~360MB, which maens that if my model is bigger than ~1.1GB, we cannot say it has learned something just that it has stored all the information in it's weights. If however the model size is smaller than ~360MB, model would have learned to go through compressed data. So this is a unique metric to look at.

Eg. GPT-2 big was ~1.75Bn Parameters trained on 40GB of uncompressed text, even if I assume ~20 GB after standard compression (would have been more than this), the model still learned a lookup table through the compressed data.


## Credits

I scraped the links from [pgnmentor.com](https://www.pgnmentor.com/files.html) and the games are by the players. I am not aware of the legality here, so if you have an problem, just raise an issue with the appropriate tag.