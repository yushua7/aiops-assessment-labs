#!/usr/bin/env python3
import json
import pathlib
import re

import spacy
from spacy.util import load_config

PATTERNS = [
    {"label": "IP", "pattern": [{"TEXT": {"REGEX": r"^(?:\d{1,3}\.){3}\d{1,3}$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^E(?:CONNREFUSED|CONNRESET|TIMEDOUT|HOSTUNREACH|NOSPC|IO|ACCES|PIPE)$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}-[0-9][0-9A-Z]{1,6}$"}}]},
]
assert PATTERNS, "add your patterns"

SRC = "models/ner/model-best"
DST = "models/ner-hybrid"
orig = load_config(f"{SRC}/config.cfg", interpolate=True)   # fully resolved values

nlp = spacy.load(SRC)
if "tok2vec" in nlp.pipe_names:
    nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])   # ner now owns a copy of the encoder
    nlp.remove_pipe("tok2vec")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(PATTERNS)
nlp.to_disk(DST)

# replace_listeners leaves ${components.tok2vec....} references in the saved config; resolve them to literal values
cfg_path = pathlib.Path(DST) / "config.cfg"
text = cfg_path.read_text()


def resolve(m):
    node = orig
    for part in m.group(1).split("."):
        node = node[part]
    return json.dumps(node)


patched, n = re.subn(r"\$\{([^}]+)\}", resolve, text)
cfg_path.write_text(patched)
print(f"patched {n} config reference(s)")

check = spacy.load(DST)
print("pipeline:", check.pipe_names, "| patterns:", len(check.get_pipe("entity_ruler").patterns))
