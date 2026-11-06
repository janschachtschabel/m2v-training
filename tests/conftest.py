import gzip
import csv
from pathlib import Path

import pytest

# Column names exactly as they appear in the WLO export.
TITLE = "properties.cclom:title"
DESCRIPTION = "properties.cclom:general_description"
KEYWORDS = "properties.cclom:general_keyword"
LANGUAGE = "properties.cclom:general_language"
SUBJECT_IDS = "properties.ccm:taxonid"
SUBJECT_NAMES = "properties.ccm:taxonid_DISPLAYNAME"
NODE_ID = "properties.sys:node-uuid"
UNUSED = "properties.ccm:price"

HEADER = [UNUSED, TITLE, DESCRIPTION, KEYWORDS, LANGUAGE, SUBJECT_IDS, SUBJECT_NAMES, NODE_ID]


def write_wlo_export(path: Path, rows: list[dict[str, str]], gz: bool = True) -> Path:
    """Write rows in the WLO export format (semicolon separated, properties.* columns)."""
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})
    return path


def make_row(
    title: str,
    description: str = "",
    keywords: str = "",
    language: str = "de",
    subject_ids: str = "",
    subject_names: str = "",
    node_id: str = "",
) -> dict[str, str]:
    return {
        TITLE: title,
        DESCRIPTION: description,
        KEYWORDS: keywords,
        LANGUAGE: language,
        SUBJECT_IDS: subject_ids,
        SUBJECT_NAMES: subject_names,
        NODE_ID: node_id or title,
        UNUSED: "kostenlos",
    }


@pytest.fixture
def export_path(tmp_path: Path) -> Path:
    rows = [
        make_row(
            "Bruchrechnung einfach erklärt",
            "Wie man Brüche addiert. Mehr unter https://example.org/brueche",
            "Bruchrechnung, Brüche, Erklärvideo",
            "de_DE",
            "http://w3id.org/openeduhub/vocabs/discipline/380",
            "Mathematik",
        ),
        make_row(
            "Photosynthese",
            "Pflanzen wandeln Licht in Energie um.",
            "Photosynthese; Biologie",
            "Deutsch",
            "http://w3id.org/openeduhub/vocabs/discipline/080",
            "Biologie",
        ),
        make_row("Photosynthese", "Pflanzen wandeln Licht in Energie um.", "Photosynthese", "de"),
        make_row("Normal force in an elevator", "How the normal force changes.", "Physics", "en"),
        make_row("Ohne Sprache", "Die Schülerinnen lernen die Gleichungen.", "", ""),
        make_row("", "Beschreibung ohne Titel", "", "de"),
    ]
    return write_wlo_export(tmp_path / "export.csv.gz", rows)


# A tiny random BERT teacher with a cased WordPiece tokenizer, so distillation and training
# can be tested offline in seconds. Its embeddings carry no meaning; tests only check mechanics.
TEACHER_WORDS = ["die", "der", "und", "Bruch", "Brüche", "Photosynthese", "Pflanze", "Satz", "des", "Pythagoras", "Licht"]


@pytest.fixture(scope="session")
def tiny_teacher(tmp_path_factory) -> str:
    return build_tiny_teacher(tmp_path_factory.mktemp("teacher"))


def build_tiny_teacher(path: Path) -> str:
    """A random 32-dimensional BERT with a cased WordPiece tokenizer, saved to `path` (no download needed)."""
    import string

    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    chars = list(dict.fromkeys(string.ascii_letters + string.digits + "äöüÄÖÜß.,;:!?-'\"()"))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *(f"##{c}" for c in chars), *TEACHER_WORDS]
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False, handle_chinese_chars=False)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(path)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    BertModel(config).save_pretrained(path)
    return str(path)


REMOTE_MODELING = '''
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class RemoteConfig(PretrainedConfig):
    model_type = "m2vtrain-test-remote"

    def __init__(self, vocab_size=100, hidden_size=32, max_position_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings


class RemoteModel(PreTrainedModel):
    """Like gte-multilingual: position ids and rotary frequencies are computed in __init__, not stored."""

    config_class = RemoteConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings), persistent=False)
        steps = torch.arange(0, config.hidden_size, 2).float() / config.hidden_size
        self.register_buffer("inv_freq", 1.0 / (10000**steps), persistent=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, input_ids, attention_mask=None, **kwargs):
        positions = self.position_ids[: input_ids.shape[1]]
        hidden = self.embeddings(input_ids) + self.position_embeddings(positions) * self.inv_freq[0]
        return BaseModelOutput(last_hidden_state=hidden)
'''


@pytest.fixture(scope="session")
def remote_teacher(tmp_path_factory, tiny_teacher) -> str:
    """A remote-code teacher (trust_remote_code) with non-persistent buffers, stored in float16 like gte."""
    import shutil
    import sys

    import torch
    from transformers import AutoTokenizer

    path = tmp_path_factory.mktemp("remote_teacher")
    (path / "modeling_remote.py").write_text(REMOTE_MODELING, encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(tiny_teacher)
    tokenizer.save_pretrained(path)
    sys.path.insert(0, str(path))
    try:
        from modeling_remote import RemoteConfig, RemoteModel

        config = RemoteConfig(
            vocab_size=len(tokenizer),
            auto_map={"AutoConfig": "modeling_remote.RemoteConfig", "AutoModel": "modeling_remote.RemoteModel"},
        )
        torch.manual_seed(0)
        RemoteModel(config).half().save_pretrained(path)
    finally:  # later loads must go through trust_remote_code, not this import
        sys.path.remove(str(path))
        sys.modules.pop("modeling_remote", None)
    shutil.rmtree(path / "__pycache__", ignore_errors=True)
    return str(path)


SYNTHETIC_TOPICS = [
    ("Bruch", "Mathematik", "https://d-nb.info/gnd/bruch"),
    ("Licht", "Physik", "https://d-nb.info/gnd/licht"),
    ("Pflanze", "Biologie", "https://d-nb.info/gnd/pflanze"),
]


def write_synthetic_dataset(folder: Path, records: int = 60) -> Path:
    """A tiny copy of the synthetic dataset's layout: data/<split>-00000-of-00001.parquet with the same columns."""
    import pandas as pd

    rows = []
    for i in range(records):
        word, subject, topic_uri = SYNTHETIC_TOPICS[i % len(SYNTHETIC_TOPICS)]
        other = TEACHER_WORDS[i % len(TEACHER_WORDS)]
        topic = f"{word} Thema {i % 12}"
        rows.append(
            {
                "id": f"schule-{i:04d}",
                "education_sector": "school",
                "language": "de",
                "title": f"{word} und {other} Material {i}",
                "description": f"Die {word} der {other} im Unterricht, Abschnitt {i}.",
                "keywords": [word, other, "Erklärvideo"],
                "fulltext": f"# {word}\n\nDie **{word}** und der {other}.\n\n- Punkt eins zu {word}\n- Punkt zwei {i}",
                "qa_pairs": [
                    {"question": f"Was ist {word} {i}?", "answer": f"{word} ist {other}.", "type": "erinnern"},
                    {"question": f"Wie hängen {word} und {other} zusammen?", "answer": "Eng.", "type": "verstehen"},
                ],
                "subject": [subject],
                "topic": topic,
                "topic_uri": f"{topic_uri}-{i % 12}",
                "generator_model": "test",
            }
        )
    frame = pd.DataFrame(rows)
    data = Path(folder) / "data"
    data.mkdir(parents=True, exist_ok=True)
    split = {"train": frame.iloc[: records - 10], "validation": frame.iloc[records - 10 : records - 5], "test": frame.iloc[records - 5 :]}
    for name, part in split.items():
        part.to_parquet(data / f"{name}-00000-of-00001.parquet", index=False)
    return Path(folder)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    return write_synthetic_dataset(tmp_path / "synthetic")
