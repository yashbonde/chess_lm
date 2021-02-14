import io
import os
import re
import json
from hashlib import md5
from chess import pgn
from tqdm import trange

# loading the information already saved
# this is our special moves handler
m2id = json.load(open('chess_lm/assets/moves.json', "r"))
results = {
    "1-0": "1",
    '1/2-1/2': "0",
    "0-1": "-1",
    '1/2 1/2': "0"
}


def parse_and_save_data(fpath, flag, save_after = 1000000):
  print_size_only = flag
  seqs = []
  rseq = []
  cntr = 0
  fcntr = 0
  hash_ = md5(fpath.encode("utf-8")).hexdigest()
  # files_pbar = trange(len(files), ncols = 100)
  game_count = 0
  game_count_loaded = 0

  # files_pbar.set_description(f"Opening file: {fpath}")
  with open(fpath, 'r', encoding = "latin") as f:
    games = f.read()
  games2 = re.sub(r"\[.*\]\s", "", games)
  game_strings = games2.split("\n\n")
  game_strings = [re.sub(r"\n", " ", gs) for gs in game_strings]
  game_count += len(game_strings)

  if print_size_only:
    print(fpath, "-->", game_count)
    return
  # gs_pbar = trange(len(game_strings), ncols = 100)
  for j in trange(len(game_strings)):
    try:
      # gs_pbar.set_description(f"Processing game: #{j}")
      gs = game_strings[j].strip()
      try:
        cg = pgn.read_game(io.StringIO(gs.decode("utf-8")))
      except:
        try:
          cg = pgn.read_game(io.StringIO(gs))
        except:
          print(f"❌ could not load game #{cntr}")
          continue
      cntr += 1 # update main counter
      result = cg.headers["Result"]
      if result != "*":
        seq = [m2id[str(x.move)[:4]] for x in cg.mainline() if str(x.move)[:4] in m2id]
        seqs.append(' '.join(list(map(str, seq))))
        rseq.append(results[result])

    except Exception as e:
      print(f"↗️ Can't open because: {e}")

    if cntr % save_after == 0:
      lm_file = f"data/{hash_}_lm_{fcntr}.txt"
      res_file = f"data/{hash_}_res_{fcntr}.txt"
      with open(lm_file, "w") as m, open(res_file, "w") as r:
        print("Saving Files...", lm_file, res_file)
        for _s, _r in zip(seqs, rseq):
          m.write(_s + "\n")
          r.write(str(_r) + "\n")
        # m.write("\n".join(seqs))
        # r.write("\n".join(list(map(str, rseq))))

      game_count_loaded += len(seqs)
      seqs = []
      rseq = []
      fcntr += 1

  print("---- Removing file ...", fpath)
  os.remove(fpath)

  if len(seqs):
    lm_file = f"data/{hash_}_lm_{fcntr}.txt"
    res_file = f"data/{hash_}_res_{fcntr}.txt"
    with open(lm_file, "w") as m, open(res_file, "w") as r:
      print("Saving Files...", lm_file, res_file)
      for _s, _r in zip(seqs, rseq):
        m.write(_s + "\n")
        r.write(str(_r) + "\n")
      # m.write("\n".join(seqs))
      # r.write("\n".join(list(map(str, rseq))))


    game_count_loaded += len(seqs)
    seqs = []
    rseq = []
    fcntr += 1

  print(f"Process Done!")

import sys
parse_and_save_data(sys.argv[1], False)
