from m2vtrain.config import RunConfig
from m2vtrain.export import metrics_table, write_model_card

RESULTS = {
    "model": {"raw": {"keyword_ndcg@10": 0.5, "subject_macro_f1": 0.25}},
    "baselines": {"tfidf": {"raw": {"keyword_ndcg@10": 0.4}}},
}


def test_metrics_table_has_one_row_per_encoder_and_mode():
    table = metrics_table(RESULTS).splitlines()
    assert len(table) == 4
    assert table[2].startswith("| this model | raw | 0.500 |")
    assert table[3].startswith("| tfidf | raw | 0.400 |") and "–" in table[3]


def test_model_card_links_hub_teacher_as_base_model(tmp_path):
    config = RunConfig(name="m2v-test", data_files=["a.csv"])
    card = write_model_card(tmp_path, config, RESULTS, {"train_documents": 10}).read_text(encoding="utf-8")

    assert card.startswith("---\n") and "\nlibrary_name: model2vec\n" in card
    assert "base_model: Alibaba-NLP/gte-multilingual-base" in card
    assert (tmp_path / "training_config.json").exists() and (tmp_path / "eval_results.json").exists()


def test_model_card_hides_data_facts_and_local_file_names_by_default(tmp_path):
    import json

    config = RunConfig.from_dict({"name": "m", "data_files": ["data/private_export.csv.gz"], "keyword_lists": {"files": ["data/kw.txt"]}})
    results = {**RESULTS, "facts": {"train_documents": 10}, "test_split": {"documents": 3, "fingerprint": "abc"}}
    card = write_model_card(tmp_path, config, results, {"train_documents": 10}).read_text(encoding="utf-8")

    assert "train_documents" not in card and "private_export" not in card
    assert "keywords (terms only" in card
    published = (tmp_path / "training_config.json").read_text(encoding="utf-8")
    assert "private_export" not in published and "kw.txt" not in published
    stored = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert "facts" not in stored and stored["test_split"] == {"fingerprint": "abc"}


def test_model_card_can_show_data_facts(tmp_path):
    config = RunConfig.from_dict({"name": "m", "data_files": ["a.csv"], "card": {"show_data_facts": True}})
    card = write_model_card(tmp_path, config, RESULTS, {"train_documents": 10}).read_text(encoding="utf-8")
    assert "train_documents: 10" in card


def test_model_card_names_the_synthetic_dataset_and_shows_cv_spread(tmp_path):
    config = RunConfig.from_dict({"name": "m", "synthetic": {"enabled": True}, "cross_validation": {"folds": 3}})
    results = {
        "model": {"raw": {"keyword_ndcg@10": 0.5, "question_mrr@10": 0.4}},
        "baselines": {},
        "cv": {"folds": 3, "std": {"model": {"raw": {"keyword_ndcg@10": 0.02, "question_mrr@10": 0.01}}, "baselines": {}}, "values": {"model": {}, "baselines": {}}},
    }
    card = write_model_card(tmp_path, config, results, {}).read_text(encoding="utf-8")

    assert "synthetic-german-educational-materials-50k" in card and "CC BY 4.0" in card
    assert "- JanSchachtschabel/synthetic-german-educational-materials-50k" in card  # datasets: in the front matter
    assert "0.500 ± 0.020" in card and "3-fold cross-validation" in card


def test_model_card_omits_local_teacher_paths(tmp_path):
    config = RunConfig.from_dict({"name": "x", "data_files": ["a.csv"], "teacher": {"model_name": str(tmp_path)}})
    assert "base_model" not in write_model_card(tmp_path, config, RESULTS, {}).read_text(encoding="utf-8")


class FakeHubApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs["repo_id"], kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs["repo_id"], kwargs))


class FailingHubApi(FakeHubApi):
    def create_repo(self, repo_id, **kwargs):
        raise OSError("not logged in")


def test_upload_model_fills_in_the_repo_id_and_uploads_privately_by_default(tmp_path):
    from m2vtrain.export import upload_model

    config = RunConfig(name="m2v-test", data_files=["a.csv"])
    write_model_card(tmp_path, config, RESULTS, {})
    (tmp_path / "model.safetensors").write_bytes(b"x")
    api = FakeHubApi()

    url = upload_model(tmp_path, "jan/m2v-test", api=api)

    assert url == "https://huggingface.co/jan/m2v-test"
    (create, repo, create_args), (upload, _, upload_args), (card, _, card_args) = api.calls
    assert (create, repo, create_args["private"]) == ("create_repo", "jan/m2v-test", True)
    assert upload == "upload_folder" and upload_args["folder_path"] == str(tmp_path)
    assert "README.md" in upload_args["ignore_patterns"]
    assert card == "upload_file" and card_args["path_in_repo"] == "README.md"
    assert b'StaticModel.from_pretrained("jan/m2v-test")' in card_args["path_or_fileobj"]
    # the stored card keeps its placeholder, so a typo or failed upload never ends up in a later upload
    assert "<user>/<model>" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_failed_upload_leaves_the_model_folder_unchanged(tmp_path):
    import pytest

    from m2vtrain.export import upload_model

    write_model_card(tmp_path, RunConfig(name="m", data_files=["a.csv"]), RESULTS, {})
    (tmp_path / "model.safetensors").write_bytes(b"x")
    before = (tmp_path / "README.md").read_text(encoding="utf-8")
    with pytest.raises(OSError):
        upload_model(tmp_path, "jan/tippfehler", api=FailingHubApi())
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == before


def test_upload_model_requires_a_model(tmp_path):
    import pytest

    from m2vtrain.export import upload_model

    with pytest.raises(FileNotFoundError):
        upload_model(tmp_path, "jan/x", api=FakeHubApi())


def test_model_card_follows_the_config(tmp_path):
    config = RunConfig.from_dict(
        {"name": "x", "data_files": ["a.csv"], "input_mode": "raw", "vocab": {"add_lemmas": False}, "data": {"languages": ["de", "no"]}}
    )
    card = write_model_card(tmp_path, config, RESULTS, {}).read_text(encoding="utf-8")
    assert "lemma share" not in card
    assert "base forms" not in card.split("## Evaluation")[0]
    assert "- 'no'" in card  # quoted, otherwise YAML reads it as false


def test_windows_teacher_paths_are_not_hub_names(tmp_path):
    from m2vtrain.export import _is_hub_name

    assert _is_hub_name("Alibaba-NLP/gte-multilingual-base")
    assert not _is_hub_name("C:/teacher")
    assert not _is_hub_name(r"models\\teacher")
