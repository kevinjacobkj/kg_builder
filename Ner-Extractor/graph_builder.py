import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def node_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{name}"


def load_json(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def entity_nodes(entities: list[dict]) -> dict[str, dict]:
    return {
        node_id(e["type"], e["name"]): {
            "id": node_id(e["type"], e["name"]),
            "type": e["type"],
            "name": e["name"],
            "score": e["score"],
            "mentions": e["mentions"],
            "chunks": list(e["chunks"]),
        }
        for e in entities
    }


def _add_chunk(chunks: list[str], chunk_id: str) -> None:
    if chunk_id not in chunks:
        chunks.append(chunk_id)


def build_graph(entities: list[dict], relationships: list[dict]) -> dict[str, list[dict]]:
    """Entities become nodes; relationship endpoints missing from `entities` get placeholder nodes (mentions 0).
    Edges are merged on (source, relation, target) across chunks with the best score."""
    nodes = entity_nodes(entities)
    extra: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    for chunk in relationships:
        for rel in chunk["relationships"]:
            ends = []
            for end in (rel["head"], rel["tail"]):
                end_id = node_id(end["type"], end["name"])
                if end_id not in nodes:
                    node = extra.setdefault(
                        end_id,
                        {"id": end_id, "type": end["type"], "name": end["name"], "score": None, "mentions": 0, "chunks": []},
                    )
                    _add_chunk(node["chunks"], chunk["id"])
                ends.append(end_id)
            key = (ends[0], rel["relation"], ends[1])
            edge = edges.setdefault(
                key, {"source": ends[0], "target": ends[1], "relation": rel["relation"], "score": rel["score"], "chunks": []}
            )
            edge["score"] = max(edge["score"], rel["score"])
            _add_chunk(edge["chunks"], chunk["id"])
    return {"nodes": list(nodes.values()) + list(extra.values()), "edges": list(edges.values())}


def write_graph(graph: dict[str, list[dict]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    entities_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "ner-entities.json"
    relationships_path = sys.argv[2] if len(sys.argv) > 2 else HERE / "ner-relationships.json"
    output_path = sys.argv[3] if len(sys.argv) > 3 else HERE / "entity-relation.json"

    graph = build_graph(load_json(entities_path), load_json(relationships_path))
    write_graph(graph, output_path)
    print(f"Wrote {len(graph['nodes'])} nodes and {len(graph['edges'])} edges to {output_path}")
