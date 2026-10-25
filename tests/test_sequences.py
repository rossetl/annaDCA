import numpy as np
import torch
from annarbm.utils.sequences import get_tokens, encode_sequence, decode_sequence, compute_weights


def test_tokens_and_encoding_roundtrip():
    assert get_tokens("protein") == "-ACDEFGHIKLMNPQRSTVWY"
    assert get_tokens("rna") == "-ACGU"
    assert get_tokens("XYZ") == "XYZ"
    sequences = ["AC-GU", "UUGCA"]
    encoded = encode_sequence(sequences, "-ACGU")
    assert encoded.shape == (2, 5)
    assert decode_sequence(encoded, "-ACGU") == sequences
    one_hot = torch.nn.functional.one_hot(torch.tensor(encoded), 5)
    assert decode_sequence(one_hot, "-ACGU") == sequences
    assert decode_sequence(encoded[0], "-ACGU") == sequences[0]


def test_compute_weights_matches_brute_force():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 3, size=(300, 10))
    data[100:150] = data[0]  # a cluster of identical sequences
    expected = np.array([1.0 / np.sum((s == data).mean(axis=1) > 0.8) for s in data])
    weights = compute_weights(data, th=0.8, chunk_size=64)
    assert np.allclose(weights.numpy(), expected)
    one_hot = torch.nn.functional.one_hot(torch.tensor(data), 3)
    assert torch.allclose(compute_weights(one_hot, th=0.8), weights)
