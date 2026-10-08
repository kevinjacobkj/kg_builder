import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple
from labels import LABELS 

# gliner calls torch.jit.script at import time and passes the deprecated `resume_download` to snapshot_download.
warnings.filterwarnings("ignore", message=".*`torch.jit.script` is deprecated")
warnings.filterwarnings("ignore", message=".*`resume_download` argument is deprecated")

from gliner import GLiNER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_files.text_parser import CHUNKS_PATH, Chunk, load_chunks  # noqa: E402
from coref import Entity, Mention, build_coref_model, predict_clusters, resolve_entities  # noqa: E402

GLINER_MODEL = "gliner-community/gliner_small-v2.5"
# GLiNER is zero-shot: keys are the prompts it scores spans against, values are the output types.
LABELS = LABELS
# Above the 0.5 default: drops generic phrases ("financial services organizations") and bare department names.
THRESHOLD = 0.7
# GLiNER truncates inputs past its word limit; this also keeps DeBERTa under 512 subwords with the label prompt.
MAX_WORDS = 384
DOC_SEPARATOR = "\n\n"

_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def build_ner_model(model_name: str = GLINER_MODEL, device: str = "cpu") -> GLiNER:
    return GLiNER.from_pretrained(model_name).to(device)


def split_text(text: str, max_words: int = MAX_WORDS) -> list[tuple[int, str]]:
    """Pack whole sentences into segments of at most `max_words` words, returning (offset, segment) pairs."""
    bounds = [0] + [m.end() for m in _SENTENCE_BREAK.finditer(text)] + [len(text)]
    segments: list[tuple[int, str]] = []
    start, words = 0, 0
    for sent_start, sent_end in zip(bounds, bounds[1:]):
        sent_words = len(_WORD.findall(text, sent_start, sent_end))
        if words and words + sent_words > max_words:
            segments.append((start, text[start:sent_start]))
            start, words = sent_start, 0
        words += sent_words
    segments.append((start, text[start:]))
    return segments


def predict(text: str, model: GLiNER) -> list[dict]:
    """GLiNER entities for `text` with offsets relative to it, typed with the PER/ORG/LOC labels."""
    entities = []
    for offset, segment in split_text(text):
        for ent in model.predict_entities(segment, list(LABELS), threshold=THRESHOLD):
            entities.append({**ent, "type": LABELS[ent["label"]], "start": ent["start"] + offset, "end": ent["end"] + offset})
    return entities


def group_by_source(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by source document, using the id prefix before the final '_<n>'."""
    groups: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.id.rsplit("_", 1)[0]].append(chunk)
    return dict(groups)


def join_chunks(chunks: list[Chunk]) -> tuple[str, list[int]]:
    """Rebuild the document text and return each chunk's character offset into it."""
    offsets, position = [], 0
    for chunk in chunks:
        offsets.append(position)
        position += len(chunk.text) + len(DOC_SEPARATOR)
    return DOC_SEPARATOR.join(chunk.text for chunk in chunks), offsets


def ner_mentions(chunks: list[Chunk], offsets: list[int], text: str, model: GLiNER) -> list[Mention]:
    mentions: list[Mention] = []
    for chunk, offset in zip(chunks, offsets):
        for ent in predict(chunk.text, model):
            start, end = ent["start"] + offset, ent["end"] + offset
            mentions.append(Mention(ent["type"], text[start:end].strip(), float(ent["score"]), start, end))
    return mentions


def entities_by_chunk(
    chunks: list[Chunk], offsets: list[int], entities: list[Entity], mentions: list[Mention]
) -> dict[str, list[dict]]:
    """One row per entity per chunk; chunks reached only through coref use the entity's best NER score."""
    scores = {(m.start, m.end): m.score for m in mentions}
    result: dict[str, list[dict]] = {}
    for chunk, offset in zip(chunks, offsets):
        rows = []
        for entity in entities:
            spans = [s for s in entity.spans if offset <= s[0] < offset + len(chunk.text)]
            if not spans:
                continue
            chunk_scores = [scores[s] for s in spans if s in entity.ner_spans]
            score = max(chunk_scores) if chunk_scores else entity.score
            rows.append((spans[0], {"entity_group": entity.type, "word": entity.name, "score": score, "count": len(spans)}))
        result[chunk.id] = [row for _, row in sorted(rows, key=lambda r: r[0])]
    return result


class Document(NamedTuple):
    chunks: list[Chunk]
    offsets: list[int]
    entities: list[Entity]
    mentions: list[Mention]


def resolve_documents(chunks: list[Chunk], ner_model: GLiNER, coref_model) -> list[Document]:
    """Run NER per chunk and coref per source document, returning each document's resolved entities."""
    groups = group_by_source(chunks)
    documents = [join_chunks(group) for group in groups.values()]
    clusters = predict_clusters([text for text, _ in documents], coref_model)

    resolved = []
    for group, (text, offsets), doc_clusters in zip(groups.values(), documents, clusters):
        mentions = ner_mentions(group, offsets, text, ner_model)
        resolved.append(Document(group, offsets, resolve_entities(text, mentions, doc_clusters), mentions))
    return resolved


def extract_entities(chunks: list[Chunk], ner_model: GLiNER, coref_model) -> dict[str, list[dict]]:
    entities: dict[str, list[dict]] = {}
    for doc in resolve_documents(chunks, ner_model, coref_model):
        entities.update(entities_by_chunk(doc.chunks, doc.offsets, doc.entities, doc.mentions))
    return {chunk.id: entities[chunk.id] for chunk in chunks}


def merge_entities(entities: dict[str, list[dict]]) -> list[dict]:
    """One row per (type, name) across chunks, in order of first appearance: max score, summed mentions."""
    merged: dict[tuple[str, str], dict] = {}
    for chunk_id, results in entities.items():
        for ent in results:
            key = (ent["entity_group"], ent["word"])
            if key not in merged:
                merged[key] = {"type": key[0], "name": key[1], "score": ent["score"], "mentions": 0, "chunks": []}
            row = merged[key]
            row["score"] = max(row["score"], ent["score"])
            row["mentions"] += ent["count"]
            if chunk_id not in row["chunks"]:
                row["chunks"].append(chunk_id)
    return [{**row, "score": round(row["score"], 4)} for row in merged.values()]


def write_entities(entities: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "ner-entities.json"

    chunks = load_chunks(input_path)
    entities = extract_entities(chunks, build_ner_model(), build_coref_model())
    merged = merge_entities(entities)
    write_entities(merged, output_path)
    print(f"Wrote {len(merged)} unique entities from {len(chunks)} chunks to {output_path}")
