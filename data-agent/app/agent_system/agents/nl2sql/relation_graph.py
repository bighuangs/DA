"""Metadata-owned relationship closure for NL2SQL retrieval and planning.

The retrieval routes intentionally return a small set of business columns.  A
small set is useful for prompt quality, but it is not necessarily a connected
SQL graph: the model may see two dimensions without the fact bridge, or it may
see the business columns while both sides of the join key were trimmed.  This
module closes that gap from the *current workspace metadata* only.

No evaluation question, reference SQL, reference-project code or table-specific
join registry is used here.  Edges are admitted only when metadata exposes a
same-name primary-key/foreign-key pair (with a conservative ``*_id`` fallback
for auto-synchronised schemas).  Ambiguous shortest paths are reported instead
of guessed.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


RELATION_CLOSURE_VERSION = "metadata-relation-closure-v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


@dataclass(frozen=True)
class RelationEdge:
    """One metadata-proved many-to-one relationship edge."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    relationship: str
    evidence: str
    confidence: float

    @property
    def signature(self) -> tuple[str, str, str, str]:
        return (
            self.left_table.lower(),
            self.left_column.lower(),
            self.right_table.lower(),
            self.right_column.lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_tables(
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for raw_table in table_infos or []:
        table = dict(raw_table)
        name = _text(table.get("name"))
        if not name:
            continue
        columns: dict[str, dict[str, Any]] = {}
        for raw_column in table.get("columns") or []:
            column = dict(raw_column)
            column_name = _text(column.get("name"))
            if column_name:
                columns[column_name.lower()] = column
        tables[name.lower()] = {
            "name": name,
            "role": _lower(table.get("role")),
            "columns": columns,
        }
    return tables


def build_relation_edges(
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> list[RelationEdge]:
    """Build safe relationship edges from live table/column metadata.

    Explicit ``primary_key`` + ``foreign_key`` is strongest.  Schema sync can
    only infer some roles, so a non-primary ``*_id`` column may act as the
    foreign side when a unique primary key with the same name exists.
    """

    tables = _metadata_tables(table_infos)
    primary_by_name: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    foreign_by_name: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    for table_name, table in tables.items():
        for column_name, column in table["columns"].items():
            role = _lower(column.get("role"))
            if role == "primary_key":
                primary_by_name.setdefault(column_name, []).append(
                    (table_name, column)
                )
            elif role == "foreign_key":
                foreign_by_name.setdefault(column_name, []).append(
                    (table_name, column, "pk_fk_role")
                )
            elif column_name.endswith("_id"):
                foreign_by_name.setdefault(column_name, []).append(
                    (table_name, column, "pk_id_convention")
                )

    edges: dict[tuple[str, str, str, str], RelationEdge] = {}
    for column_name, foreign_rows in foreign_by_name.items():
        primary_rows = primary_by_name.get(column_name, [])
        for foreign_table, foreign_column, evidence in foreign_rows:
            for primary_table, primary_column in primary_rows:
                if foreign_table == primary_table:
                    continue
                edge = RelationEdge(
                    left_table=tables[foreign_table]["name"],
                    left_column=_text(foreign_column.get("name")),
                    right_table=tables[primary_table]["name"],
                    right_column=_text(primary_column.get("name")),
                    relationship="many_to_one",
                    evidence=evidence,
                    confidence=1.0 if evidence == "pk_fk_role" else 0.9,
                )
                edges[edge.signature] = edge
    return sorted(edges.values(), key=lambda item: item.signature)


def _adjacency(edges: Sequence[RelationEdge]) -> dict[str, list[RelationEdge]]:
    output: dict[str, list[RelationEdge]] = {}
    for edge in edges:
        output.setdefault(edge.left_table.lower(), []).append(edge)
        output.setdefault(edge.right_table.lower(), []).append(edge)
    for rows in output.values():
        rows.sort(key=lambda item: item.signature)
    return output


def _other(edge: RelationEdge, table_name: str) -> str:
    if edge.left_table.lower() == table_name:
        return edge.right_table.lower()
    return edge.left_table.lower()


def _shortest_paths(
    starts: set[str],
    target: str,
    adjacency: Mapping[str, Sequence[RelationEdge]],
    *,
    max_hops: int,
) -> list[list[RelationEdge]]:
    """Return a bounded set of shortest simple paths."""

    queue: deque[tuple[str, list[RelationEdge], set[str]]] = deque(
        (start, [], {start}) for start in sorted(starts)
    )
    shortest: int | None = None
    paths: list[list[RelationEdge]] = []
    while queue:
        node, path, visited = queue.popleft()
        if shortest is not None and len(path) >= shortest:
            continue
        if len(path) >= max_hops:
            continue
        for edge in adjacency.get(node, ()):
            next_node = _other(edge, node)
            if next_node in visited:
                continue
            next_path = [*path, edge]
            if next_node == target:
                shortest = len(next_path) if shortest is None else shortest
                if len(next_path) == shortest and len(paths) < 16:
                    paths.append(next_path)
                continue
            queue.append((next_node, next_path, {*visited, next_node}))
    return paths


def _path_score(
    path: Sequence[RelationEdge],
    tables: Mapping[str, Mapping[str, Any]],
) -> tuple[int, float, int]:
    nodes = {
        name
        for edge in path
        for name in (edge.left_table.lower(), edge.right_table.lower())
    }
    fact_nodes = sum(
        str(tables.get(name, {}).get("role") or "") in {"fact", "fact_table"}
        for name in nodes
    )
    return (
        len(path),
        -round(sum(edge.confidence for edge in path), 6),
        -fact_nodes,
    )


def close_relation_requirements(
    table_infos: Sequence[Mapping[str, Any]] | None,
    required_columns: Mapping[str, Sequence[str] | set[str]] | None,
    *,
    max_hops: int = 4,
    max_tables: int = 8,
) -> dict[str, Any]:
    """Add unique bridge tables and join keys to a required-column mapping.

    The function is deliberately fail-closed.  If two paths have the same
    length, evidence strength and fact-table preference, neither is guessed;
    the target is returned in ``unresolved_tables`` for the model/clarification
    path to handle.
    """

    tables = _metadata_tables(table_infos)
    canonical_required: dict[str, set[str]] = {}
    original_required: dict[str, set[str]] = {}
    unknown_tables: list[str] = []
    for raw_table, raw_columns in (required_columns or {}).items():
        table_key = _lower(raw_table)
        if table_key not in tables:
            unknown_tables.append(_text(raw_table))
            continue
        table_name = tables[table_key]["name"]
        available = tables[table_key]["columns"]
        selected = {
            _text(available[_lower(column)].get("name"))
            for column in raw_columns or []
            if _lower(column) in available
        }
        if selected:
            canonical_required.setdefault(table_name, set()).update(selected)
            original_required.setdefault(table_name, set()).update(selected)

    terminals = {name.lower() for name in canonical_required}
    edges = build_relation_edges(table_infos)
    adjacency = _adjacency(edges)
    joins: list[RelationEdge] = []
    unresolved = list(dict.fromkeys(unknown_tables))
    ambiguous: list[dict[str, Any]] = []
    connected: set[str] = set()
    if terminals:
        fact_terminals = sorted(
            name
            for name in terminals
            if str(tables.get(name, {}).get("role") or "") in {"fact", "fact_table"}
        )
        connected.add((fact_terminals or sorted(terminals))[0])

    remaining = set(terminals - connected)
    while remaining:
        best_target: str | None = None
        best_path: list[RelationEdge] | None = None
        best_score: tuple[int, float, int] | None = None
        best_tied_paths: list[list[RelationEdge]] = []
        for target in sorted(remaining):
            paths = _shortest_paths(
                connected, target, adjacency, max_hops=max(1, int(max_hops))
            )
            if not paths:
                continue
            scored = sorted(
                ((_path_score(path, tables), path) for path in paths),
                key=lambda item: (
                    item[0],
                    tuple(edge.signature for edge in item[1]),
                ),
            )
            score = scored[0][0]
            tied = [path for path_score, path in scored if path_score == score]
            if best_score is None or score < best_score:
                best_target = target
                best_path = tied[0]
                best_score = score
                best_tied_paths = tied
        if best_target is None or best_path is None:
            unresolved.extend(tables[name]["name"] for name in sorted(remaining))
            break
        signatures = {
            tuple(edge.signature for edge in path) for path in best_tied_paths
        }
        if len(signatures) > 1:
            ambiguous.append({
                "target_table": tables[best_target]["name"],
                "candidate_path_count": len(signatures),
                "hop_count": len(best_path),
            })
            unresolved.append(tables[best_target]["name"])
            remaining.remove(best_target)
            continue

        for edge in best_path:
            if edge.signature not in {item.signature for item in joins}:
                joins.append(edge)
            canonical_required.setdefault(edge.left_table, set()).add(
                edge.left_column
            )
            canonical_required.setdefault(edge.right_table, set()).add(
                edge.right_column
            )
            connected.update(
                {edge.left_table.lower(), edge.right_table.lower()}
            )
        if len(connected) > max(1, int(max_tables)):
            unresolved.extend(
                tables[name]["name"] for name in sorted(remaining)
            )
            break
        remaining = terminals - connected - {
            _lower(item.get("target_table")) for item in ambiguous
        }

    added_columns: dict[str, list[str]] = {}
    for table_name, columns in canonical_required.items():
        added = columns - original_required.get(table_name, set())
        if added:
            added_columns[table_name] = sorted(added)
    original_tables = {name.lower() for name in original_required}
    added_tables = sorted(
        name for name in canonical_required if name.lower() not in original_tables
    )
    return {
        "schema_version": RELATION_CLOSURE_VERSION,
        "status": (
            "closed" if not unresolved and not ambiguous else "partial"
        ),
        "required_columns": {
            table: sorted(columns)
            for table, columns in sorted(canonical_required.items())
        },
        "joins": [edge.to_dict() for edge in joins],
        "terminal_tables": sorted(
            tables[name]["name"] for name in terminals if name in tables
        ),
        "added_tables": added_tables,
        "added_columns": added_columns,
        "unresolved_tables": sorted(set(unresolved)),
        "ambiguous_paths": ambiguous,
        "edge_count": len(edges),
    }


def compact_relation_closure(
    closure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the bounded relationship proof shared by SSE and persisted Trace.

    Only authorised schema identifiers, relationship types and counts survive.
    Raw questions, retrieved documents, vectors and filter values are never
    accepted by this whitelist.
    """

    value = dict(closure or {})

    def names(items: Any, *, limit: int = 16) -> list[str]:
        return [str(item)[:128] for item in list(items or [])[:limit]]

    required_columns = {
        str(table)[:128]: names(columns, limit=24)
        for table, columns in list(
            dict(value.get("required_columns") or {}).items()
        )[:16]
    }
    added_columns = {
        str(table)[:128]: names(columns, limit=16)
        for table, columns in list(
            dict(value.get("added_columns") or {}).items()
        )[:16]
    }
    joins = []
    for raw in list(value.get("joins") or [])[:16]:
        if not isinstance(raw, Mapping):
            continue
        joins.append({
            key: raw.get(key)
            for key in (
                "left_table", "left_column", "right_table", "right_column",
                "relationship", "evidence", "confidence",
            )
            if raw.get(key) is not None
        })
    ambiguous = []
    for raw in list(value.get("ambiguous_paths") or [])[:8]:
        if not isinstance(raw, Mapping):
            continue
        ambiguous.append({
            key: raw.get(key)
            for key in ("target_table", "candidate_path_count", "hop_count")
            if raw.get(key) is not None
        })
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "required_columns": required_columns,
        "terminal_tables": names(value.get("terminal_tables")),
        "added_tables": names(value.get("added_tables")),
        "added_columns": added_columns,
        "joins": joins,
        "unresolved_tables": names(value.get("unresolved_tables")),
        "ambiguous_paths": ambiguous,
        "edge_count": int(value.get("edge_count") or 0),
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }
