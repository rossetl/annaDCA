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
    save_chains(path, chains, tokens="01", continuous_labels=True)
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


def _write_dataset(path, num=60, length=8, missing=False):
    rng = np.random.default_rng(0)
    sequences = ["".join(rng.choice(list("ACDE"), size=length)) for _ in range(num)]
    labels = rng.choice(["a", "b"], size=num).astype(object)
    if missing:
        labels[::4] = None
    pd.DataFrame({"name": [f"s{i}" for i in range(num)], "sequence": sequences, "label": labels}).to_csv(path, index=False)


def test_cli_train_overwrite_log_and_sample(tmp_path):
    import pytest
    from annadca.cli import main
    from annadca.io import load_chains
    from annadca.rbm import load_rbm
    path = tmp_path / "data.csv"
    _write_dataset(path, missing=True)
    out = tmp_path / "out"
    common = ["train", "-d", str(path), "-o", str(out), "-l", "run", "--device", "cpu", "--nchains", "30",
              "--batch_size", "16", "-H", "5", "--save_every", "5", "--no_compile", "--alphabet", "ACDE",
              "--lr_final", "0.001", "--max_grad_norm", "10", "--l2_label", "0.01"]
    main(common + ["--nupdates", "10"])
    with pytest.raises(FileExistsError):
        main(common + ["--nupdates", "10"])
    main(common + ["--nupdates", "10", "--overwrite"])

    ckpt_dir = out / "run_checkpoints"
    main(common + ["--nupdates", "15", "-p", str(ckpt_dir / "model_update_010.pt")])
    log = (out / "run.log").read_text()
    assert "Resumed from" in log and "Pearson Cij" in log
    rows = [line.split() for line in log.splitlines() if line.split() and line.split()[0].isdigit()]
    assert [int(r[0]) for r in rows] == [0, 5, 10, 15]
    assert all(np.isfinite(float(r[3])) for r in rows)

    rbm, checkpoint = load_rbm(str(ckpt_dir / "model_update_015.pt"))
    assert checkpoint["metadata"]["idx_to_label"] == {0: "a", 1: "b"}
    assert checkpoint["metadata"]["args"]["pseudocount"] > 0  # set to 1/Meff
    # the cosine schedule also follows the total number of updates after resuming
    from annadca.scripts.train import learning_rate
    assert np.isclose(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], learning_rate(14, 0.01, 0.001, 15))
    assert np.isclose(learning_rate(15, 0.01, 0.001, 15), 0.001) and np.isclose(learning_rate(0, 0.01, 0.001, 15), 0.01)

    samples = tmp_path / "samples.fasta"
    main(["sample", "-p", str(ckpt_dir / "model_update_015.pt"), "-o", str(samples), "-n", "12",
          "--gibbs_steps", "5", "--target", "b", "--device", "cpu"])
    chains = load_chains(str(samples), tokens="ACDE", num_classes=2, label_to_idx={"a": 0, "b": 1})
    assert chains["visible"].shape == (12, 8, 4)
    assert torch.all(chains["label"].argmax(-1) == 1)
