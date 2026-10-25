import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from annadca.rbm import get_rbm
from annadca.io import save_chains, load_chains
from annadca.utils.analysis import mutual_information

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_save_load_chains_categorical(tmp_path):
    rbm = get_rbm("potts", "bernoulli", (6, 4), 3, 3)
    chains = rbm.init_chains(10)
    chains["label"][0] = 0.0  # missing label
    path = str(tmp_path / "chains.fasta")
    save_chains(path, chains, tokens="ACGU", idx_to_label={0: "x", 1: "y", 2: "z"})
    loaded = load_chains(path, tokens="ACGU", num_classes=3, label_to_idx={"x": 0, "y": 1, "z": 2})
    assert torch.equal(loaded["visible"], chains["visible"])
    assert torch.equal(loaded["label"], chains["label"])


def test_save_load_chains_binary_continuous(tmp_path):
    rbm = get_rbm("bernoulli", "bernoulli", 6, 3, 1, continuous_label=True)
    chains = rbm.init_chains(10)
    path = str(tmp_path / "chains.fasta")
    save_chains(path, chains, tokens="01")
    loaded = load_chains(path, tokens="01", num_classes=1, continuous_labels=True)
    assert torch.equal(loaded["visible"], chains["visible"])
    assert torch.allclose(loaded["label"], chains["label"], rtol=1e-5)


def test_mutual_information_shapes():
    rbm = get_rbm("potts", "bernoulli", (6, 4), 3, 3)
    chains = rbm.init_chains(100)
    assert mutual_information(chains["visible"], chains["label"]).shape == (3, 6, 4)
    rbm = get_rbm("bernoulli", "bernoulli", 6, 3, 3)
    chains = rbm.init_chains(100)
    assert mutual_information(chains["visible"], chains["label"]).shape == (3, 6)


def _train(*args):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "annadca", "scripts", "train.py"), *args],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_train_and_resume(tmp_path):
    rng = np.random.default_rng(0)
    sequences = ["".join(rng.choice(list("ACDE"), size=8)) for _ in range(60)]
    path = tmp_path / "data.csv"
    pd.DataFrame({"name": [f"s{i}" for i in range(60)], "sequence": sequences, "label": rng.choice(["a", "b"], size=60)}).to_csv(path, index=False)
    out = tmp_path / "out"
    common = ["-d", str(path), "-o", str(out), "-l", "test", "--device", "cpu", "--nchains", "20", "-H", "5",
              "--save_every", "10", "--no_compile", "--init_from_data", "--alphabet", "ACDE"]
    _train(*common, "--nepochs", "20")
    checkpoint_dir = out / "test_checkpoints"
    assert sorted(os.listdir(checkpoint_dir)) == ["model_update_000.pt", "model_update_010.pt", "model_update_020.pt"]

    _train(*common, "--nepochs", "30", "-p", str(checkpoint_dir / "model_update_020.pt"))
    checkpoint = torch.load(checkpoint_dir / "model_update_030.pt", map_location="cpu")
    assert checkpoint["update"] == 30
    rbm = get_rbm("potts", "bernoulli", (8, 4), 5, 2)
    rbm.load_state_dict(checkpoint["model_state_dict"])
    assert all(torch.isfinite(p).all() for p in rbm.parameters())
