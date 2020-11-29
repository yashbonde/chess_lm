# Chess LM

Current neural chess engines like AlphaZero train a network based on self-play trajectories. Now that is a bit too much for me to handle alone and neither do I have the compute necessary. So it is possible to train a supervised model on millions of games and play against it.

Read [blog](https://yashbonde.github.io/blogs/chess-lm.html) for more information on the project.

<!-- I cannot believe that people do [such](https://arxiv.org/pdf/2008.04057.pdf) garbage work and can get
away with a paper. This is just sad man. -->

Now added is the code to server and a simple webapge to play in:
```
python3 play.py # goto http://0.0.0.0:5000/ in browser
```

To see a random gameplay between two AI agents run `python3 game.py` which will print value, confidence for each move and create a PGN file. To see the gameplay online goto this [website](https://chesstempo.com/pgn-viewer/), copy pase your PGN.


## It Learns!...?

Somehow there are signs that it is infact good at predicting the value of the board, if not at the policy.
Consider this simple game where two networks with the same weights were playing against one another. One used the minimax tree searching, while the other was using sampling.

<img src="assets/self-play1.png">

Note that the game did not turn out to be like that, infact both the models play (policy) very stupidly. However they do correctly understand the state of the board and balck better than white because of that depth search. This used a `z5` model and 

## Player

The way to evaluate the model is to make it into a player and run it. To make a player do the following:
```python
from game import GameEngine, Player

with open("assets/moves.json") as f:
    vocab_size = len(json.load(f))

# config for model
config = ModelConfig(vocab_size = vocab_size, n_positions=60, n_ctx=60, n_embd=128, n_layer=30, n_head=8)

# make two player
player1 = Player(config, ".model_sample/z4_0.pt", "assets/moves.json")
player2 = Player(config, ".model_sample/z4_0.pt", "assets/moves.json")

# define game engine
game = GameEngine()

# play
mv = 0
done = False
p = 0
while not done:
    # model returns move object, value, confidence of move
    if p == 0:
        m,v,c = player1.move(game)
        p += 1
    else:
        m,v,c = player2.move(game)
        p = 0
    done, res = game.step(m)
    print(m,v,c)
```

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

Or skip this entire part and download the ZIP using the following command:

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tdUgOB1VOnIT6opEJBzptp_rfUSEcbUZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tdUgOB1VOnIT6opEJBzptp_rfUSEcbUZ" -O agg.zip && rm -rf /tmp/cookies.txt
unzip agg.zip
```

This will download a zip file and extract to different files `agg_mv.txt` and `agg_res.txt` which are the moves file and results file respectively. (Read Credits for further clarification)

### Training

The if you have unzipped in same level as this repo then training is straightforward, run the command
```
python train.py --model=<model-name>
```
I use 2x1080Ti configuration with 128 GB of RAM, `batch_size=350` seems to fill just about both the GPUs. Model `z5` has the baseline config.

### Training Logs

| name | n_embd | n_layer | buffer_size | batch_size | maxlen |
| ---- | ------ | ------- | ----------- | ---------- | ------ |
| v0   | 128    | 30      | 55555       | 350        | 60     |
| v6   | 256    | 20      | 1000000     | 256        | 60     |
| z5   | 128    | 30      | Full        | 350        | 60     |
| q1   | 128    | 30      | Full        | 90         | 180    |

Consider the loss graph below, grey one is `v0`, red is `v6` and orange is `z5`. You can see that larger buffer improves the training as seen between `v0` and `v6`, both in overall loss and smoother loss curves. When compared with fully loaded dataset in `z5` the loss curve is more smoother while the training takes longer. It eventually does reach the lower loss value (epoch-end). Due to a bug in the `IterableDataset` number of samples was lower than fully loaded counterpart also seen is that a larger model gives only a slight edge over the smaller counterpart while parameters are ~3x.

<img src="assets/loss_f.png">

These curves obey the neural language models scaling laws as proposed by OpenAI. Read more in this [blog](https://yashbonde.github.io/blogs/chess-lm-1_5.html).

<!-- |name|win_pred|acc_pred|
|-|-|-|
|z5_0.pt|0.0183%|0%| -->

## Todo

This is the task list:

- [ ] Use `gdown` to download from google drive
- [ ] Perform Accuracy run on different models and log (in progress)
  - Create a small subset of data for testing: create a conditional in `download.py` which runs like `python3 download.py -s 5` and creates a smaller file with top 5% of all the sequences in ZIP extract. Remember to do it for both the `agg_moves.txt` and `agg_res.txt`.
  - Write a python script to evaluate different models on this small subset, it takes in a txt file which has list of paths to different checkpoints. This is done because in the final version all the models will have to be evaluated automatically on the complete dataset and there should be no human in the loop.
- [ ] Integrate code with a web interface on [chessshhh](https://github.com/yashbonde/chessshhh)
  - ~~Make a neural player~~
  - Port code for server
    - There's this CORS shit for API and static file code is not being read correctly. Someone else has to fix those.
- [ ] Add Simple MinMax Tree search
  - ~~Write code for minimax~~
  - Integrate with `Player`
- [ ] Add more complicated search algorithm MCTS

## Credits

The code I have written is under MIT License, code used under its own licenses. I scraped the links from [pgnmentor.com](https://www.pgnmentor.com/files.html) and the games are by the players. I am not aware of the legality here, so if you have an problem, just raise an issue. For now I have uploaded the parsed ZIP file on my GDrive.
