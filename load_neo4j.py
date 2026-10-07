import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

ROOT = Path(__file__).resolve().parent

NODE_LABELS = {
    "PER": "Person",
    "TITLE": "JobTitle",
    "ORG": "Organization",
    "COMP": "Company",
    "DEPT": "Department",
    "TEAM": "Team",
    "BU": "BusinessUnit",
    "PROD": "Product",
    "SOFT": "Software",
    "PLAT": "Platform",
    "TECH": "Technology",
    "SERV": "Service",
    "APP": "Application",
    "DB": "Database",
    "LOC": "Location",
    "CITY": "City",
    "STATE": "State",
    "COUNTRY": "Country",
    "REGION": "Region",
    "PROJECT": "Project",
    "PROGRAM": "Program",
    "INITIATIVE": "Initiative",
    "CONTRACT": "Contract",
    "AGREEMENT": "Agreement",
    "FUNCTION": "Function",
    "PROCESS": "Process",
    "EVENT": "Event",
    "DATE": "Date",
    "MONEY": "Money",
    "PERCENT": "Percentage",
    "EMAIL": "Email",
    "PHONE": "Phone",
    "URL": "Url",
}
_NOT_IDENT = re.compile(r"[^A-Za-z0-9_]+")

CONSTRAINT = "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
NODE_QUERY = """
UNWIND $rows AS row
MERGE (n:Entity {id: row.id})
SET n:`%s`, n.name = row.name, n.type = row.type, n.score = row.score, n.mentions = row.mentions, n.chunks = row.chunks
"""
EDGE_QUERY = """
UNWIND $rows AS row
MATCH (a:Entity {id: row.source})
MATCH (b:Entity {id: row.target})
MERGE (a)-[r:`%s`]->(b)
SET r.relation = row.relation, r.score = row.score, r.chunks = row.chunks
"""
# CALL ... IN TRANSACTIONS needs an auto-commit transaction (session.run), not execute_query.
DELETE_ALL = "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _identifier(text: str, fallback: str) -> str:
    return _NOT_IDENT.sub("_", text).strip("_") or fallback


def node_label(entity_type: str) -> str:
    return NODE_LABELS.get(entity_type) or _identifier(entity_type, "Unknown")


def relationship_type(relation: str) -> str:
    """'works at' -> 'WORKS_AT'."""
    return _identifier(relation, "RELATED_TO").upper()


def group_by(rows: list[dict], key) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def _names(driver: Driver, query: str, database: str) -> list[str]:
    return [record["name"] for record in driver.execute_query(query, database_=database).records]


def _count(driver: Driver, query: str, database: str) -> int:
    return driver.execute_query(query, database_=database).records[0][0]


def wipe_database(driver: Driver, database: str) -> tuple[int, int, int, int]:
    """Drop every constraint and non-LOOKUP index, then delete all data; returns (constraints, indexes, nodes, relationships)."""
    constraints = _names(driver, "SHOW CONSTRAINTS YIELD name", database)
    for name in constraints:
        driver.execute_query(f"DROP CONSTRAINT {_quote(name)} IF EXISTS", database_=database)
    indexes = _names(driver, "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name", database)
    for name in indexes:
        driver.execute_query(f"DROP INDEX {_quote(name)} IF EXISTS", database_=database)
    nodes = _count(driver, "MATCH (n) RETURN count(n)", database)
    relationships = _count(driver, "MATCH ()-[r]->() RETURN count(r)", database)
    with driver.session(database=database) as session:
        session.run(DELETE_ALL).consume()
    return len(constraints), len(indexes), nodes, relationships


def load_graph(driver: Driver, graph: dict[str, list[dict]], database: str) -> tuple[int, int]:
    """MERGE nodes per label and edges per relationship type; returns (nodes created, relationships created)."""
    driver.execute_query(CONSTRAINT, database_=database)
    nodes_created = edges_created = 0
    for label, rows in group_by(graph["nodes"], lambda n: node_label(n["type"])).items():
        summary = driver.execute_query(NODE_QUERY % label, rows=rows, database_=database).summary
        nodes_created += summary.counters.nodes_created
    for rel_type, rows in group_by(graph["edges"], lambda e: relationship_type(e["relation"])).items():
        summary = driver.execute_query(EDGE_QUERY % rel_type, rows=rows, database_=database).summary
        edges_created += summary.counters.relationships_created
    return nodes_created, edges_created


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    graph_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Ner-Extractor" / "entity-relation.json"

    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        sys.exit("NEO4J_PASSWORD is not set; add it to .env (and NEO4J_URI / NEO4J_USERNAME / NEO4J_DATABASE if not the defaults).")
    if database.lower() == "system":
        sys.exit("Refusing to overwrite the 'system' database; set NEO4J_DATABASE to a data database.")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            driver.verify_connectivity()
            print(f"Overwriting database '{database}' at {uri}")
            constraints, indexes, old_nodes, old_edges = wipe_database(driver, database)
            print(f"Dropped {constraints} constraints and {indexes} indexes; deleted {old_nodes} nodes and {old_edges} relationships")
            nodes_created, edges_created = load_graph(driver, graph, database)
    except AuthError:
        sys.exit(f"Neo4j authentication failed for user '{username}' at {uri}; check NEO4J_USERNAME / NEO4J_PASSWORD in .env.")
    except ServiceUnavailable as exc:
        sys.exit(f"Neo4j is not reachable at {uri}: {exc}")
    print(f"Loaded {nodes_created} nodes and {edges_created} relationships into '{database}'")
