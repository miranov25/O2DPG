"""PHASE_13_75_ADF fixture generator (seeded, regenerable). Produces:
  lazy_struct_fixture_clean.root   — happy-path fixture (no hazards beyond jagged member)
  lazy_struct_fixture_hazard.root  — adds collision decoy (H3 negative control)
  chain_part1.root / chain_part2.root — clean twins for T-AUTO-2 chain tests
Requires uproot>=5 (mktree; NOTE: dict-assignment writes RNTuple, NOT TTree)."""
import uproot, awkward as ak, numpy as np

def _payload(rng, N):
    return {
        "dedxTPC/dEdxMaxTPC": rng.normal(50,10,N).astype(np.float32),
        "dedxTPC/dEdxTotTPC": rng.normal(40,8,N).astype(np.float32),
        "dedxTPC/dEdxMaxIROC": rng.normal(30,5,N).astype(np.float32),
        "dedxTPC/clusterQ": ak.Array([rng.normal(10,2,int(k)).astype(np.float32).tolist()
                                      for k in rng.integers(0,6,N)]),
        "mTOFLength/len": rng.normal(370,10,N).astype(np.float32),
        "jag": ak.Array([rng.normal(0,1,int(k)).astype(np.float32).tolist()
                         for k in rng.integers(0,6,N)]),
        "mult": rng.integers(0,100,N).astype(np.int32),
        "tgl": rng.uniform(-1,1,N).astype(np.float32),
        "decoyUnused": rng.normal(0,1,N),
    }

_TYPES = {"dedxTPC/dEdxMaxTPC": np.float32, "dedxTPC/dEdxTotTPC": np.float32,
          "dedxTPC/dEdxMaxIROC": np.float32, "dedxTPC/clusterQ": "var * float32",
          "mTOFLength/len": np.float32, "jag": "var * float32",
          "mult": np.int32, "tgl": np.float32, "decoyUnused": np.float64}

def make(path, seed, N=200, hazard=False):
    rng = np.random.default_rng(seed)
    types = dict(_TYPES); data = _payload(rng, N)
    if hazard:
        types["dEdxMaxTPC__dedxTPC"] = np.float32
        data["dEdxMaxTPC__dedxTPC"] = rng.normal(0,1,N).astype(np.float32)
    with uproot.recreate(path) as f:
        f.mktree("tree", types); f["tree"].extend(data)
    return path

def make_all(d="."):
    make(f"{d}/lazy_struct_fixture_clean.root", 7)
    make(f"{d}/lazy_struct_fixture_hazard.root", 7, hazard=True)
    make(f"{d}/chain_part1.root", 11, N=120)
    make(f"{d}/chain_part2.root", 13, N=80)

if __name__ == "__main__":
    make_all()
