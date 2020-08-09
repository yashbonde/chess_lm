"""perform pre-processing on the file and bring all content to same place"""

from glob import glob

agg_mv = []
agg_res = []
print(glob("lms/*.txt"))
for idx in range(int(len(glob("lms/*.txt")) / 2 - 2)):
    with open(f"lms/chess_res_{idx}.txt") as f:
        agg_res.extend(f.read().split())
    with open(f"lms/chess_lm_{idx}.txt") as f:
        agg_mv.extend(f.readlines())

assert (
    len(agg_mv) == len(agg_res),
    f"results and games should be same, but found {len(agg_mv)} games and {len(agg_res)} results"
)

with open(f"lms/agg_mv.txt", "w") as f1, open(f"lms/agg_res.txt", "w") as f2:
    f1.write("\n".join([x.strip() for x in agg_mv]))
    f2.write("\n".join(agg_res))
