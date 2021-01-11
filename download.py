"""download and save the data - @yashbonde"""
import io
import os
import re
import sys
import h5py
import json
import wget
import time
import chess
import pickle
import subprocess
import numpy as np
import multiprocessing
from chess import pgn
from glob import glob
from tqdm import trange
from zipfile import ZipFile


def verbose_print(*args, verbose):
    # print only when verbose is True
    if verbose:
        print(*args)


# loading the information already saved
m2id = json.load(open('assets/moves.json', "r"))  # this is our special moves handler
results = {
    "1-0": "1",
    '1/2-1/2': "0",
    "0-1": "-1",
    '1/2 1/2': "0"
}

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


def parse_and_save_data(files, save_after, pid, verbose = False):
    """parse the pgn files and save the data after `save_after` no. of games have been processed
    """
    verbose_print(pid, "----", files, verbose = verbose)
    seqs = []
    rseq = []
    cntr = 0
    fcntr = 0
    # files_pbar = trange(len(files), ncols = 100)
    game_count = 0
    game_count_loaded = 0
    for i in range(len(files)):
        verbose_print(pid, "-- Opening -->>>", files[i], verbose = verbose)
        fpath = files[i]
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
                        verbose_print(f"❌ could not load game #{cntr}", verbose = verbose)
                        continue
                cntr += 1 # update main counter
                result = cg.headers["Result"]
                if result != "*":
                    seq = [m2id[str(x.move)[:4]] for x in cg.mainline() if str(x.move)[:4] in m2id]
                    seqs.append(' '.join(list(map(str, seq))))
                    rseq.append(results[result])

            except Exception as e:
                verbose_print(f"↗️ Can't open because: {e}", verbose = verbose)

            if cntr % save_after == 0:
                lm_file = f"data/chess_lm_{fcntr}_{pid}.txt"
                res_file = f"data/chess_res_{fcntr}_{pid}.txt"
                with open(lm_file, "w") as m, open(res_file, "w") as r:
                    verbose_print("Saving Files...", lm_file, res_file, verbose = verbose)
                    m.write("\n".join(seqs))
                    r.write("\n".join(list(map(str, rseq))))

                game_count_loaded += len(seqs)
                seqs = []
                rseq = []
                fcntr += 1

        verbose_print(pid, "---- Removing file ...", fpath, verbose = verbose)
        os.remove(fpath)

    if len(seqs):
        lm_file = f"data/chess_lm_{fcntr}_{pid}.txt"
        res_file = f"data/chess_res_{fcntr}_{pid}.txt"
        with open(lm_file, "w") as m, open(res_file, "w") as r:
            verbose_print("Saving Files...", lm_file, res_file, verbose = verbose)
            m.write("\n".join(seqs))
            r.write("\n".join(list(map(str, rseq))))


        game_count_loaded += len(seqs)
        seqs = []
        rseq = []
        fcntr += 1

    verbose_print(f"Process: {pid} Done!", verbose = verbose)


def parse_and_save_data_2(files, save_after, pid, verbose = False):
    """parse the pgn files and save the data after `save_after` no. of games have been processed,
    There may have been a bug with the other one. Not sure if it was bug because of the function
    or parsing.
    """
    print(":: Starting Process:", pid)
    seqs = []
    rseq = []
    cntr = 0
    fcntr = 0
    game_count = 0
    game_count_loaded = 0
    for i in range(len(files)):
        fpath = files[i]
        with open(fpath, 'r', encoding = "latin") as f:
            games = f.read()
        games2 = re.sub(r"\[.*\]\s", "", games)
        game_strings = games2.split("\n\n")
        game_strings = [re.sub(r"\n", " ", gs) for gs in game_strings]
        game_count += len(game_strings)
        
        for gs in game_strings:
            try:
                cntr = 0
                gs = gs.strip()
                try:
                    cg = pgn.read_game(io.StringIO(gs.decode("utf-8")))
                except:
                    try:
                        cg = pgn.read_game(io.StringIO(gs))
                    except:
                        print(f"❌ could not load game #{cntr}", verbose = verbose)
                        continue
                res = cg.headers["Result"]
                if res != "*":
                    cntr += 1 # update main counter
                    seq = [m2id[str(x.move)[:4]] for x in cg.mainline()]
                    seqs.append(seq)
                    rseq.append(results[res])
            except Exception as e:
                print(f"↗️ Can't open because: {e}")
                continue

            if cntr % save_after == 0:
                pkl_file = f"data/chess_lm_{fcntr}_{pid}.p"
                with open(pkl_file, "wb") as p:
                    print("Saving File...", pkl_file)
                    pickle.dump({"lms": seqs, "res": rseq}, p)

                game_count_loaded += len(seqs)
                seqs = []
                rseq = []
                fcntr += 1

    if len(seqs):
        pkl_file = f"data/chess_lm_{fcntr}_{pid}.p"
        with open(pkl_file, "wb") as p:
            print("Saving File...", pkl_file)
            pickle.dump({"lms": seqs, "res": rseq}, p)
        game_count_loaded += len(seqs)
        
    print(f"Process: {pid} Done!")



def leela_dataset_compiler(files, save_after, pid, save_legal_mask=False, verbose=False, _trange = False):
    """parse the pgn files and save the data after `save_after` no. of games have been processed.
    This function is specifically built to parse leelachess self play games. which have the following
    format:
    1.Nf3 d5 2.g3 Nf6 3.Bg2 c5 4.d4 cxd4 5.O-O e6 6.Nxd4 Be7 7.c4 O-O 8.cxd5 exd5 9.Be3 Nc6 10.Qd3
    Ng4 11.Rd1 Nxe3 12.Bxd5 0-1 {OL: 0}

    If you need to understand how the files are written check here:
    https://github.com/LeelaChessZero/lczero-client/blob/e3bdd17c5a96ddc4a6c6caab8dcfd68ea2aa620d/lc0_main.go#L237
    """
    verbose_print(":::: initialising PID:", pid, verbose = verbose)
    seqs = []
    rseq = []
    # legals = []
    fcntr = 0
    game_count_loaded = 0
    pbar = trange(len(files)) if _trange else range(len(files))
    for i in pbar:
        with open(files[i], "r") as f:
            game_str = f.read()
        cg = pgn.read_game(io.StringIO(game_str))
        if "0-1" in game_str:
            res = -1
        elif "1-0" in game_str:
            res = 1
        elif "1/2-1/2" in game_str:
            res = 0
        else:
            continue

        # now convert this game to out moves and create legal masking
        seq = []
        if save_legal_mask:
            legal_mask = []
            b = chess.Board()
            for x in cg.mainline():
                mv = m2id[str(x.move)[:4]]
                b.push(x.move)

                # store data in bool to save memory (1793 bytes vs 14344 bytes) use array.nbytes
                # using bitarray is even smaller (just 239 bytes) but it does not work on vast.ai's
                # Docker python
                legal = np.zeros(shape=(len(m2id))).astype(np.bool)
                legal[[m2id[str(x)[:4]] for x in b.legal_moves]] = True

                seq.append(mv)
                legal_mask.append(legal)
        
        else:
            # can optimise code here
            seq = [m2id[str(x.move)[:4]] for x in cg.mainline()]

        # add to master 
        rseq.append(res)
        seqs.append(seq)
        # legals.append(legal_mask)

        if i and i % save_after == 0:
            # this time not writing things as txt files store data as pickle
            pkl_path = f"data/chess_{fcntr}_{pid}.p"
            print(f"Save data at {pkl_path}")
            with open(pkl_path, "wb") as f:
                # pickle.dump({"lms": seqs, "res": rseq, "msk": legals}, f)
                pickle.dump({"lms": seqs, "res": rseq}, f)

            game_count_loaded += len(seqs)
            seqs = []
            rseq = []
            fcntr += 1

    if len(seqs):
        # this time not writing things as txt files store data as pickle
        pkl_path = f"data/chess_{fcntr}_{pid}.p"
        print(f"Save data at {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump({"lms": seqs, "res": rseq, "msk": legals}, f)

        game_count_loaded += len(seqs)
        seqs = []
        rseq = []
        fcntr += 1

    verbose_print(f"Process: {pid} Done! Loaded: {game_count_loaded} Games!", verbose = verbose)


def multiprocessing_parsing_wrapper(files, save_after, leela = False):
    WORKERS = min(20, len(files))
    process_list = []
    
    files_bkts = [[] for _ in range(WORKERS)]
    for i,f in enumerate(files):
        i = i % WORKERS # to put in proper bucket
        files_bkts[i].append(f)

    func = parse_and_save_data_2 if not leela else leela_dataset_compiler
    for pid in range(WORKERS):
        process_list.append(multiprocessing.Process(
            target = func,
            args = (files_bkts[pid], save_after, pid,)
        ))
        process_list[-1].start()

    for p in process_list:
        p.join()


def print_help():
    h = '''
Chess LM sata preparation script. Many things in this script are
    hardcoded and you might need to open it and change those.
Usage: download.py [-hdpmc] [options]
-h     : Display this help and exit
-d     : Download the data from assets/lc0.txt file
         Extract the ZIP files and dumps them in data/ folder
-p     : Parse the .pgn files in data/ folder using multprocessing
         default workers set at 20. After parsing process for each
         worker you wil get a .txt file with the move ids
-m     : Merges all the parsed files to a single giant txt file
-c m f : Compiles the merge txt file to a big dataset. Takes
         following arguments:
         - <m:int> maximum length for each game
         - <f:str ["npz","hdf5"]> format to store data in
-l     : Builds Leela Chess Dataset for files in data/ folder
'''.strip()
    print(h)

# ----------------------------------------------- #
if sys.argv[1] == "-h":
    print_help()

elif sys.argv[1] == "-d":
    # download files and unzip
    links = open('assets/lc0.txt').readlines()
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
    all_files = [x for x in glob("./data/chess_lm*.p")]
    print("Found", len(all_files), "files!")

    # we create exhaustive list, this will also help in determining the dataloading procedure to use
    all_sequences = []
    all_results = []

    # now we open files one by one
    total_moves = []
    print(":::::: First Pass ::::::")
    for i, name in enumerate(all_files):
        print("Reading file...", name)
        with open(name, "rb") as p:
            data = pickle.load(p)
            lm = data["lms"]
            res = data["res"]
            # uncomment below to see the number of moves
            total_moves.append(sum([len(x) for x in lm]))
            all_sequences.extend(lm)
            all_results.extend(res)
        print(f"Total Games: {len(all_sequences)}; Moves: {sum(total_moves)}")
        print("-"*80)
        
        if i and i % 10 == 0:
            print(f"Writing ---> data/all_target_{i}.p")
            with open(f"data/all_target_{i}.p", "wb") as p:
                pickle.dump({"lms":all_sequences, "res":all_results}, p)
            all_sequences = []
            all_results = []
            total_moves = []

    # second pass to merge even more
    print(":::::: Second Pass ::::::")
    all_files = [x for x in glob("./data/all_target*.p")]
    for i, name in enumerate(all_files):
        print("Reading file...", name)
        with open(name, "rb") as p:
            data = pickle.load(p)
            lm = data["lms"]
            res = data["res"]
            # uncomment below to see the number of moves
            total_moves.append(sum([len(x) for x in lm]))
            all_sequences.extend(lm)
            all_results.extend(res)
    
    print(":: total games:", len(all_sequences), "total_moves:", sum(total_moves))

    print("Writing ---> data/final.p")
    with open("data/final.p", "wb") as p:
        pickle.dump({"lms":all_sequences, "res":all_results}, p)


elif sys.argv[1] == "-c":
    # compiles the dataset into HDF5 or NPZ
    maxlen = int(sys.argv[2])
    format = sys.argv[3]

    # compile to .hdf5 file
    len_file = 0
    with open("data/all_lm.txt", "r") as f:
        for _ in f:
            len_file += 1
    total_len = len_file
    GAME = 0

    # now load the complete dataset in memory
    with open("data/all_lm.txt", "r") as flm, open("data/all_res.txt", "r") as fres:
        lms = []  # all the sequences
        results = []  # all the results
        print("Loading samples in memory ... this will take some time")
        for idx, lm, game_res in zip(trange(total_len), flm, fres):
            # ignore BOS + EOS tags, [GAME] does it for us
            lm = list(map(lambda x: int(x.strip()), lm.split()))[1:-1]
            lms.extend([GAME] + lm)

            # get the targets for values as [0,res,-res,res,-res...]
            game_res = float(game_res)
            res = np.ones(len(lm)) * game_res
            res[1:len(lm):2] = -game_res
            results.extend([0] + res.tolist())  # first will always generate 0
    
    # convert to lms and results
    lms = np.array(lms[:-(len(lms) % maxlen)]).reshape(-1, maxlen)
    results = np.array(results[:-(len(results) % maxlen)]).reshape(-1, maxlen)

    if format == "hdf5":
        print("Saving HDF5 at data/chessD2.hdf5")
        hdf = h5py.File("data/chessD2.hdf5", "w")

        hdf.create_dataset("lms", shape=lms.shape, dtype="i", data=lms)
        hdf.create_dataset("res", shape=results.shape, dtype="i", data=results)

        hdf.close()

    elif format == "npz":
        print("Saving Numpy zip at data/clm.npz")
        np.savez("data/clm.npz", lms=lms, res=results)


elif sys.argv[1] == "-l":
    # this stores the leela chess dump
    parsing_start_time = time.time()
    pgnfiles = glob('data/*.pgn')
    print(f"Found {len(pgnfiles)} files.")
    multiprocessing_parsing_wrapper(pgnfiles, 1000000, leela = True)
    ty_res = time.gmtime(time.time() - parsing_start_time)
    res = time.strftime("%H:%M:%S", ty_res)
    print(f"Parsing completed in {res}")

    # now we open files one by one
    all_files = glob("data/*.p")
    total_moves = 0
    all_sequences = []
    all_results = []
    for name in all_files:
        print("Reading files...", name)
        with open(name, "r") as f:
            data = pickle.load(f)
            lms = data["lms"]
            # uncomment below to see the number of moves
            total_moves += sum([len(x) for x in lms])
            all_sequences.extend(lms)
            all_results.extend(data["res"])

    # print
    print("-"*70)
    print(f"total_moves: {total_moves}")
    print(f"total_games: {len(all_sequences)}")
    print("-"*70)

    # restructure it and return
    print("Writing file data/leela.p")
    with open("data/leela.p", "wb") as f:
        pickle.dumps({
            "lms": all_sequences,
            "res": all_results
        })

else:
    print_help()
