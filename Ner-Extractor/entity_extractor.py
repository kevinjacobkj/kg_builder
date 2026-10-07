import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_files.text_parser import Chunk, load_chunks  # noqa: E402
from coref import Entity, Mention, build_coref_model, predict_clusters, resolve_entities  # noqa: E402

MODEL_NAME = "dslim/bert-base-NER"
DOC_SEPARATOR = "\n\n"


def build_ner_pipeline(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # "simple" merges word-piece tokens (e.g. "John", "##son") into whole entities.
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


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


def ner_mentions(chunks: list[Chunk], offsets: list[int], text: str, nlp) -> list[Mention]:
    mentions: list[Mention] = []
    for chunk, offset in zip(chunks, offsets):
        for ent in nlp(chunk.text):
            start, end = ent["start"] + offset, ent["end"] + offset
            mentions.append(Mention(ent["entity_group"], text[start:end].strip(), float(ent["score"]), start, end))
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


def extract_entities(chunks: list[Chunk], nlp, coref_model) -> dict[str, list[dict]]:
    groups = group_by_source(chunks)
    documents = [join_chunks(group) for group in groups.values()]
    clusters = predict_clusters([text for text, _ in documents], coref_model)

    entities: dict[str, list[dict]] = {}
    for group, (text, offsets), doc_clusters in zip(groups.values(), documents, clusters):
        mentions = ner_mentions(group, offsets, text, nlp)
        resolved = resolve_entities(text, mentions, doc_clusters)
        entities.update(entities_by_chunk(group, offsets, resolved, mentions))
    return {chunk.id: entities[chunk.id] for chunk in chunks}


def write_entities(entities: dict[str, list[dict]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for chunk_id, results in entities.items():
        lines.append(f"[{chunk_id}]")
        for ent in results:
            lines.append(f"{ent['entity_group']}\t{ent['word']}\t{ent['score']:.4f}\t{ent['count']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "chunks.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "ner-entities.txt"

    chunks = load_chunks(input_path)
    entities = extract_entities(chunks, build_ner_pipeline(), build_coref_model())
    write_entities(entities, output_path)
    total = sum(len(v) for v in entities.values())
    print(f"Wrote {total} entities from {len(chunks)} chunks to {output_path}")
