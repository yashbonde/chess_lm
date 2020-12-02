"""download and save the data - @yashbonde"""
import io
import os
import re
import sys
import json
import wget
import time
import chess
import subprocess
import multiprocessing
from chess import pgn
from glob import glob
from tqdm import trange
from zipfile import ZipFile


# loading the information already saved
m2id = json.load(open('assets/moves.json', "r"))  # this is our special moves handler
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
    print(pid, "----", files)
    seqs = []
    rseq = []
    cntr = 0
    fcntr = 0
    # files_pbar = trange(len(files), ncols = 100)
    game_count = 0
    game_count_loaded = 0
    for i in range(len(files)):
        print(pid, "-- Opening -->>>", files[i])
        fpath = files[i]
        all_lines = []
        # files_pbar.set_description(f"Opening file: {fpath}")
        with open(fpath, 'r', encoding = "latin") as f:
            games = f.read()
        games2 = re.sub(r"\[.*\]\s", "", games)
        game_strings = games2.split("\n\n")
        game_strings = [re.sub(r"\n", " ", gs) for gs in game_strings]
        game_count += len(game_strings)
        # gs_pbar = trange(len(game_strings), ncols = 100)
        for j in range(len(game_strings)):
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
                
                cntr += 1 # update main counter
                result = cg.headers["Result"]
                if result is not "*":
                    seq = [m2id[str(x.move)[:4]] for x in cg.mainline() if str(x.move)[:4] in m2id]
                    seqs.append(' '.join(list(map(str, seq))))
                    rseq.append(results[result])

            except Exception as e:
                print(f"↗️ Can't open because: {e}")

            if cntr % save_after == 0:
                lm_file = f"data/chess_lm_{fcntr}_{pid}.txt"
                res_file = f"data/chess_res_{fcntr}_{pid}.txt"
                with open(lm_file, "w") as m, open(res_file, "w") as r:
                    print("Saving Files...", lm_file, res_file)
                    m.write("\n".join(seqs))
                    r.write(" ".join(list(map(str, rseq))))

                game_count_loaded += len(seqs)
                seqs = []
                rseq = []
                fcntr += 1

        print(pid, "---- Removing file ...", fpath)
        os.remove(fpath)

    if len(seqs):
        lm_file = f"data/chess_lm_{fcntr}_{pid}.txt"
        res_file = f"data/chess_res_{fcntr}_{pid}.txt"
        with open(lm_file, "w") as m, open(res_file, "w") as r:
            print("Saving Files...", lm_file, res_file)
            m.write("\n".join(seqs))
            r.write(" ".join(list(map(str, rseq))))


        game_count_loaded += len(seqs)
        seqs = []
        rseq = []
        fcntr += 1

    print(f"Process: {pid} Done!")


def multiprocessing_parsing_wrapper(files, save_after):
    WORKERS = min(20, len(files))
    process_list = []
    
    files_bkts = [[] for _ in range(WORKERS)]
    for i,f in enumerate(files):
        i = i % WORKERS # to put in proper bucket
        files_bkts[i].append(f)

    for pid in range(WORKERS):
        process_list.append(multiprocessing.Process(
            target = parse_and_save_data,
            args = (files_bkts[pid], save_after, pid,)
        ))
        process_list[-1].start()

    for p in process_list:
        p.join()


# ----------------------------------------------- #

if sys.argv[1] == "-d":
    # download files and unzip
    links = open('assets/links2000.txt').readlines()
    os.makedirs("data/", exist_ok = True)
    download_start_time = time.time()
    zippaths = []
    print(f"Downloading {len(links)} Zip Files ...")
    pbar = trange(len(links))
    for i in pbar:
        l = links[i]
        zp = wget.download(l.strip(), "data")
        zippaths.append(zp)

    # log the time taken
    ty_res = time.gmtime(time.time() - download_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Download completed in {res}")

    # from pgnmentor
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
        
    # FICS uses .bz2 file
    print("Extracting Zip Files ...")
    zip_start_time = time.time()
    zipfiles = glob("data/*.bz2")
    print(f"No. of zip files: {len(zipfiles)}")
    for i in trange(len(zipfiles)):
        f = zipfiles[i]
        subprocess.run(["bzip2", "-d", f])

    ty_res = time.gmtime(time.time() - zip_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Unzipping completed in {res}")


elif sys.argv[1] == "-p":
    # parse files
    parsing_start_time = time.time()
    pgnfiles = glob('data/*.pgn')
    print(f"Found {len(pgnfiles)} files.")
    multiprocessing_parsing_wrapper(pgnfiles, 1000000)
    ty_res = time.gmtime(time.time() - parsing_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Parsing completed in {res}")


elif sys.argv[1] == "-m":
    # merge files and split into testing and training
    all_files = glob("./data/*.txt")
    by_order = {}
    for f in all_files:
        name = "".join(re.findall("\d+_\d+", f))
        if not name:
            continue
        mode = "res" if "res" in f else "lm"
        if name not in by_order:
            by_order[name] = {mode: f}
        else:
            by_order[name].update({mode: f})
     
    # we create exhaustive list, this will also help in determining the dataloading procedure to use
    all_sequences = []
    all_results = []
    
    # now we open files one by one
    total_moves = 0
    for name in by_order:
        print("Reading files...", name)
        with open(by_order[name]["lm"], "r") as lm, open(by_order[name]["res"], "r") as res:
            lm = lm.read()
            res = res.read().split("\n")
            # uncomment below to see the number of moves
            # total_moves += len(re.findall("\s", lm)) - len(res)
            all_sequences.extend(lm.split("\n"))
            all_results.extend(res)

    print("============= Total Games:", len(all_sequences))
    print("============= Total Moves:", total_moves)

    print("Writing ---> data/all_lm.txt, data/all_res.txt")
    with open("data/all_lm.txt", "w") as train_lm, open("data/all_res.txt", "w") as train_res:
        train_lm.write("\n".join(all_sequences))
        train_res.write("\n".join(all_results))

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
