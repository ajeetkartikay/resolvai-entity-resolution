"""Generate a small synthetic dataset in the pipeline's input format (no real data needed)."""
import os, random, sys

out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
os.makedirs(out, exist_ok=True)
random.seed(1)
WORDS = ["acme", "robotics", "delta", "foods", "bright", "cafe", "zen", "traders", "kappa", "motors",
         "fortune", "finance", "raj", "blue", "ocean", "green", "valley", "star", "metro", "prime"]
STREETS = ["Market St", "Oak Ave", "Pine St", "Hill Rd", "Lake Dr", "MG Road", "Rue Voltaire", "Bd Victor Hugo"]


def make(split, n, off):
    s1, s2, s3, gt = [], [], [], []
    for k in range(n):
        c = random.choice(["US", "India", "France"] if split == "test" else ["US", "India"])
        name = " ".join(random.sample(WORDS, 2)).title() + f" {k + off}"
        addr = f"{random.randint(1, 999)} {random.choice(STREETS)}, City{k % 40}"
        eid = f"S1-{k + off}"
        s1.append((eid, name + " Inc", addr, c))
        ms = []
        for m in range(random.choice([0, 1, 2, 3])):           # 0 matches = singleton
            src = random.choice([2, 3])
            mid = f"S{src}-{k + off}{m}"
            n2 = name.upper() if random.random() < .5 else name[:-1]            # case change / typo
            a2 = addr.replace("St", "Street") if random.random() < .5 else ""  # abbreviation / missing address
            (s2 if src == 2 else s3).append((mid, n2, a2, c))
            ms.append(mid)
        gt.append((eid, ",".join(ms)))
    s2.append((f"S2-q{off}", 'Odd "quote name', "x", "US"))     # quoting edge case
    for fn, rows in [(f"{split}_source1.tsv", s1), (f"{split}_source2.tsv", s2), (f"{split}_source3.tsv", s3)]:
        with open(os.path.join(out, fn), "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.writelines("\t".join(r) + "\n" for r in rows)
    if split == "train":
        with open(os.path.join(out, "train_ground_truth.tsv"), "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.writelines("\t".join(r) + "\n" for r in gt)


make("train", 6000, 0)
make("test", 3000, 100000)
print("synthetic data written to", out)
