import json
import sys
import warnings
from pathlib import Path
from labels import LABELS, RELATIONS

# gliner calls torch.jit.script at import time and passes the deprecated `resume_download` to snapshot_download.
warnings.filterwarnings("ignore", message=".*`torch.jit.script` is deprecated")
warnings.filterwarnings("ignore", message=".*`resume_download` argument is deprecated")

from gliner import GLiNER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_files.text_parser import CHUNKS_PATH, Chunk, load_chunks  # noqa: E402
from coref import Span, build_coref_model  # noqa: E402
from entity_extractor import Document, build_ner_model, resolve_documents, split_text  # noqa: E402

RELEX_MODEL = "knowledgator/gliner-relex-base-v1.0"
# Relex's own entities above this score are added when no NER/coref span covers them (e.g. "project").
DISCOVERY_THRESHOLD = 0.6
# Output is limited to the given spans, so a low entity threshold only lets pronouns and repeat mentions into the graph.
ENTITY_THRESHOLD = 0.3
ADJACENCY_THRESHOLD = 0.5
RELATION_THRESHOLD = 0.6
LABEL_OF = {short: label for label, short in LABELS.items()}

Node = tuple[str, str]
Triple = tuple[Node, str, Node, float]


def build_relex_model(model_name: str = RELEX_MODEL, device: str = "cpu") -> GLiNER:
    return GLiNER.from_pretrained(model_name).to(device)


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def canonical_nodes(doc: Document, index: int) -> dict[Span, Node]:
    """Chunk-relative spans of every NER and coref mention in chunk `index`, mapped to the entity's (name, type)."""
    offset, length = doc.offsets[index], len(doc.chunks[index].text)
    return {
        (start - offset, end - offset): (entity.name, entity.type)
        for entity in doc.entities
        for start, end in entity.spans
        if offset <= start and end <= offset + length
    }


def discover_nodes(text: str, known: dict[Span, Node], model: GLiNER) -> dict[Span, Node]:
    """Relex entities that overlap no known span, kept under their raw text."""
    found = model.inference([text], list(LABELS), relations=[], threshold=DISCOVERY_THRESHOLD, return_relations=False)[0]
    return {
        (ent["start"], ent["end"]): (ent["text"], LABELS[ent["label"]])
        for ent in found
        if not any(_overlap((ent["start"], ent["end"]), span) for span in known)
    }


def match_node(ent: dict, nodes: dict[Span, Node]) -> Node:
    """The node whose span overlaps the relex span most, or the raw relex text if none does."""
    span = (ent["start"], ent["end"])
    best = max(nodes, key=lambda s: _overlap(s, span), default=None)
    if best is not None and _overlap(best, span):
        return nodes[best]
    return ent["text"], LABELS.get(ent["type"], ent["type"])


def predict_relations(text: str, nodes: dict[Span, Node], model: GLiNER) -> list[Triple]:
    triples: list[Triple] = []
    for offset, segment in split_text(text):
        local = {
            (start - offset, end - offset): node
            for (start, end), node in nodes.items()
            if offset <= start and end <= offset + len(segment)
        }
        local |= discover_nodes(segment, local, model)
        if not local:
            continue
        # Only the segment's own entity types: offering the full label set sharply lowers relation scores.
        labels = sorted({LABEL_OF[entity_type] for _, entity_type in local.values() if entity_type in LABEL_OF})
        _, relations = model.inference(
            [segment],
            labels,
            relations=list(RELATIONS),
            threshold=ENTITY_THRESHOLD,
            adjacency_threshold=ADJACENCY_THRESHOLD,
            relation_threshold=RELATION_THRESHOLD,
            flat_ner=False,
            input_spans=[[{"start": start, "end": end} for start, end in sorted(local)]],
        )
        for rel in relations[0]:
            triples.append((match_node(rel["head"], local), rel["relation"], match_node(rel["tail"], local), rel["score"]))
    return triples


def _allowed(head: Node, relation: str, tail: Node) -> bool:
    head_types, tail_types = RELATIONS[relation]
    return head != tail and head[1] in head_types and tail[1] in tail_types


def relationship_rows(triples: list[Triple]) -> list[dict]:
    """Type-checked triples, deduplicated on (head, relation, tail) with the best score, highest score first."""
    best: dict[tuple[Node, str, Node], float] = {}
    for head, relation, tail, score in triples:
        if _allowed(head, relation, tail):
            key = (head, relation, tail)
            best[key] = max(best.get(key, 0.0), float(score))
    return [
        {
            "head": {"name": head[0], "type": head[1]},
            "relation": relation,
            "tail": {"name": tail[0], "type": tail[1]},
            "score": round(score, 4),
        }
        for (head, relation, tail), score in sorted(best.items(), key=lambda item: -item[1])
    ]


def extract_relationships(chunks: list[Chunk], ner_model: GLiNER, coref_model, relex_model: GLiNER) -> dict[str, list[dict]]:
    relationships: dict[str, list[dict]] = {}
    for doc in resolve_documents(chunks, ner_model, coref_model):
        for index, chunk in enumerate(doc.chunks):
            triples = predict_relations(chunk.text, canonical_nodes(doc, index), relex_model)
            relationships[chunk.id] = relationship_rows(triples)
    return {chunk.id: relationships[chunk.id] for chunk in chunks}


def write_relationships(relationships: dict[str, list[dict]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": chunk_id, "relationships": rows} for chunk_id, rows in relationships.items()]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "ner-relationships.json"

    chunks = load_chunks(input_path)
    relationships = extract_relationships(chunks, build_ner_model(), build_coref_model(), build_relex_model())
    write_relationships(relationships, output_path)
    total = sum(len(rows) for rows in relationships.values())
    print(f"Wrote {total} relationships from {len(chunks)} chunks to {output_path}")
