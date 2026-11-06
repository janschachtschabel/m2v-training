import json

import pytest

from m2vtrain.config import RunConfig, load_config, save_config


def test_default_config_roundtrips_through_json(tmp_path):
    config = RunConfig(name="test", data_files=["data/export.csv.gz"])
    path = save_config(config, tmp_path / "config.json")

    assert json.loads(path.read_text(encoding="utf-8"))["teacher"]["model_name"] == "Alibaba-NLP/gte-multilingual-base"
    assert load_config(path) == config


def test_partial_json_uses_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "x", "data_files": ["a.csv"], "pca_dims": 128}), encoding="utf-8")
    config = load_config(path)
    assert config.pca_dims == 128
    assert config.vocab.max_vocab_size == 200_000


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="input_mode"):
        RunConfig(name="x", data_files=["a.csv"], input_mode="sometimes")
    with pytest.raises(ValueError, match="data_files"):
        RunConfig(name="x", data_files=[])
    with pytest.raises(ValueError, match="eval_modes"):
        RunConfig(name="x", data_files=["a.csv"], eval_modes=["mixed"])


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "x", "data_files": ["a.csv"], "pca_dim": 128}), encoding="utf-8")
    with pytest.raises(ValueError, match="pca_dim"):
        load_config(path)


def test_partial_sections_keep_the_run_defaults():
    config = RunConfig.from_dict(
        {"name": "x", "data_files": ["a.csv"], "pairs": {"train": {"max_epochs": 2}}, "teacher": {"device": "cuda"}}
    )
    assert config.pairs.train.learning_rate == 1e-4  # PairsConfig default, not TrainOptions' 1e-3
    assert config.pairs.train.max_epochs == 2
    assert config.teacher.model_name == "Alibaba-NLP/gte-multilingual-base"
    assert config.teacher.trust_remote_code is True
    assert config.teacher.device == "cuda"


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"tokenlearn": {"enabled": "false"}}, "enabled"),
        ({"data_files": "data/x.csv"}, "data_files"),
        ({"name": 5}, "name"),
        ({"pca_dims": 1.5}, "pca_dims"),
        ({"pairs": {"max_pairs": 0}}, "max_pairs"),
        ({"tokenlearn": {"max_texts": -1}}, "max_texts"),
        ({"vocabulary_quantization": 0}, "vocabulary_quantization"),
    ],
)
def test_wrong_types_and_ranges_are_rejected(override, key):
    with pytest.raises(ValueError, match=key):
        RunConfig.from_dict({"name": "x", "data_files": ["a.csv"], **override})


def test_ints_are_accepted_for_float_options():
    assert RunConfig.from_dict({"name": "x", "data_files": ["a.csv"], "lemma_share": 1}).lemma_share == 1
