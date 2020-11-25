"""download and save the data - @yashbonde"""
import io
import os
import re
import sys
import json
import wget
import time
import requests
import chess
from chess import pgn
from glob import glob
from tqdm import trange, tqdm_notebook
from zipfile import ZipFile


# loading the information already saved
m2id = json.load(open('m2id.json', "r"))  # this is our special moves handler
results = {
    "1-0": "1",
    '1/2-1/2': "0",
    "0-1": "-1",
    '1/2 1/2': "0"
}

def seconds_to_hours(s):
    mins = s // 60
    hrs = mins // 60
    days = hrs // 24
    return days, hrs, mins


def load_game_count(files):
    files_pbar = trange(len(files), ncols = 100)
    game_count = 0
    for i in files_pbar:
        fpath = files[i]
        all_lines = []
        files_pbar.set_description(f"Opening file: {fpath}")
        with open(fpath, 'r', encoding = "latin") as f:
            games = f.read()
        games2 = re.sub(r"\[.*\]\s", "", games)
        game_strings = games2.split("\n\n")
        game_strings = [re.sub(r"\n", " ", gs) for gs in game_strings]
        game_count += len(game_strings)
    return game_count


def parse_and_save_data(files, save_after, pid):
    """parse the pgn files and save the data after `save_after` no. of games have been processed
    """
    seqs = []
    rseq = []
    cntr = 0
    fcntr = 0
    files_pbar = trange(len(files), ncols = 100)
    game_count = 0
    game_count_loaded = 0
    for i in files_pbar:
        fpath = files[i]
        all_lines = []
        files_pbar.set_description(f"Opening file: {fpath}")
        with open(fpath, 'r', encoding = "latin") as f:
            games = f.read()
        games2 = re.sub(r"\[.*\]\s", "", games)
        game_strings = games2.split("\n\n")
        game_strings = [re.sub(r"\n", " ", gs) for gs in game_strings]
        game_count += len(game_strings)
        gs_pbar = trange(len(game_strings), ncols = 100)
        for j in gs_pbar:
            try:
                gs_pbar.set_description(f"Processing game: #{j}")
                gs = game_strings[j].strip()
                try:
                    cg = pgn.read_game(io.StringIO(gs.decode("utf-8")))
                except:
                    try:
                        cg = pgn.read_game(io.StringIO(gs))
                    except:
                        print(f"❌ could not load game #{cntr}")
                
                cntr += 1 # update main counter
                result = cg.headers["Result"]
                if result is not "*":
                    seq = [m2id[str(x.move)[:4]] for x in cg.mainline() if str(x.move)[:4] in m2id]
                    seqs.append(' '.join(list(map(str, seq))))
                    rseq.append(results[result])

            except Exception as e:
                print(f"↗️ Can't open because: {e}")

            if cntr % save_after == 0:
                with open(f"chess_lm_{fcntr}_{pid}.txt", "w") as m, open(f"chess_res_{fcntr}_{pid}.txt", "w") as r:
                    print("Saving Files...")
                    m.write("\n".join(seqs))
                    r.write(" ".join(list(map(str, rseq))))

                game_count_loaded += len(seqs)
                seqs = []
                rseq = []
                fcntr += 1

        os.remove(fpath)

    if len(seqs):
        with open(f"chess_lm_{fcntr}.txt", "w") as m, open(f"chess_res_{fcntr}.txt", "w") as r:
            print("Saving Files...")
            m.write("\n".join(seqs))
            r.write(" ".join(list(map(str, rseq))))

        game_count_loaded += len(seqs)
        seqs = []
        rseq = []
        fcntr += 1    

    return game_count, game_count_loaded


# def multiprocessing_parsing_wrapper(files, save_after):





# ----------------------------------------------- #

if sys.argv[1] == "-d":
    # get the links and download using HTTP
    links = open('links.txt').readlines()

    os.makedirs("data/", exists_ok = True)
    
    download_start_time = time.time()

    zippaths = []
    print(f"Downloading {len(links)} Zip Files ...")
    pbar = trange(len(links), ncols = 200)
    for i in pbar:
        l = links[i]
        pbar.set_description(f"Downloading {l}")
        # zp = './' + l.split('/')[-1].strip()
        # if os.path.exists(zp):
        #     print(f"Skipping ... {l}")
        #     continue

        # initially I used requests but after introducin FICS database which had
        # some file > 3GB the requests would have shat the bed so now I am using
        # wget
        # data = requests.get(l.strip())
        # with open(zp, "wb") as f:
        #     f.write(data.content)

        zp = wget.download(l.strip(), "data")
        zippaths.append(zp)

    # log the time taken
    ty_res = time.gmtime(time.time() - download_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Download completed in {res}")

    print("Extracting Zip Files ...")
    zip_start_time = time.time()
    zipfiles = glob("data/*.zip")
    print(f"No. of zip files: {len(zipfiles)}")
    for i in trange(len(zipfiles)):
        f = zipfiles[i]
        zf = ZipFile(f, 'r')
        zf.extractall('.')
        zf.close()
        os.remove(f)

    ty_res = time.gmtime(time.time() - zip_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Unzipping completed in {res}")

# NOTE: that the meta information `game_count`, `game_count_loaded` was added after it was run
# please use it on small file set before running on the complete system
elif sys.argv[1] == "-p":
    parsing_start_time = time.time()
    pgnfiles = glob('data/*.pgn')
    print(f"Found {len(pgnfiles)} files.")
    game_count, game_count_loaded = parse_and_save_data(pgnfiles, 100000) # n*~36MB files
    print(f"Found {game_count} games")
    print(f"Loaded {game_count_loaded} ({game_count_loaded/game_count * 100}% coverage)")
    ty_res = time.gmtime(time.time() - parsing_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Parsing completed in {res}")

# ----------------------------------------------- #

def get_all_moves():
    """return all possible moves in the baord and map those to squares"""
    num_moves = 0
    moves = []
    # for x in range(8):
    for x, f in enumerate(chess.FILE_NAMES):
        # for y in range(8):
        for y, r in enumerate(chess.RANK_NAMES):
            # print(x, y)
            # knight moves
            kmoves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
            for mv in kmoves:
                if 7 >= x + mv[0] >=0 and 7 >= y + mv[1] >=0:
                    sq = chess.square(x + mv[0], y + mv[1])
                    moves.append("{}{}{}".format(f,r, chess.square_name(sq)))

            for i in range(8):
                # hv moves
                if i != x:
                    sq = chess.square(i, y)
                    moves.append("{}{}{}".format(f,r, chess.square_name(sq)))
                if i != y:
                    sq = chess.square(x, i)
                    moves.append("{}{}{}".format(f,r, chess.square_name(sq)))
                    
            # diag moves --> flip x, ys
            directions = list(map(list, [
                zip(range(x + 1, 8), range(y + 1, 8)),
                zip(range(x + 1, 8), range(y - 1, -1, -1)),
                zip(range(x - 1, -1, -1), range(y + 1, 8)),
                zip(range(x - 1, -1, -1), range(y - 1, -1, -1)),
            ]))

            diag_moves = []
            for d in directions:
                diag_moves.extend(d)
            for i,mv in enumerate(diag_moves):
                sq = chess.square(*mv)
                if sq == "{}{}".format(f,r):
                    continue
                diag_moves[i] = "{}{}{}".format(f,r, chess.square_name(sq))
            moves.extend(diag_moves)

    moves_idx = {mv:i for i,mv in enumerate(moves)}
    return moves_idx
