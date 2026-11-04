"""Graphical utilities for DAG-DGPs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Mapping, Sequence


UndirectedAdjacency = Dict[str, set[str]]


def topological_sort(parents: Mapping[str, Sequence[str]]) -> list[str]:
    """Return a topological ordering of the nodes."""
    nodes = list(parents.keys())
    node_set = set(nodes)

    for node, node_parents in parents.items():
        for parent in node_parents:
            if parent not in node_set:
                raise ValueError(f"Unknown parent '{parent}' referenced by node '{node}'.")

    indegree = {node: len(node_parents) for node, node_parents in parents.items()}
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for node, node_parents in parents.items():
        for parent in node_parents:
            children[parent].append(node)

    queue = [node for node in nodes if indegree[node] == 0]
    topo: list[str] = []

    while queue:
        node = queue.pop(0)
        topo.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(topo) != len(nodes):
        raise ValueError("Cycle detected in DAG.")

    return topo


def moralize_dag(
    parents: Mapping[str, Sequence[str]],
    *,
    nodes: Iterable[str] | None = None,
) -> UndirectedAdjacency:
    """Return the moral graph defined by the parent mapping."""
    if nodes is None:
        selected = tuple(parents.keys())
    else:
        selected = tuple(nodes)

    selected_set = set(selected)
    unknown = selected_set - set(parents.keys())
    if unknown:
        raise ValueError(f"Unknown nodes requested for moralization: {sorted(unknown)}.")

    graph: UndirectedAdjacency = {node: set() for node in selected}
    for child, node_parents in parents.items():
        if child not in selected_set:
            continue
        kept_parents = [parent for parent in node_parents if parent in selected_set]
        for parent in kept_parents:
            graph[child].add(parent)
            graph[parent].add(child)
        for left, right in combinations(kept_parents, 2):
            graph[left].add(right)
            graph[right].add(left)
    return graph


@dataclass(frozen=True)
class DAG:
    """A lightweight directed acyclic graph container."""

    parents: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        _ = topological_sort(self.parents)

    @property
    def nodes(self) -> list[str]:
        return list(self.parents.keys())

    @property
    def roots(self) -> list[str]:
        return [node for node, node_parents in self.parents.items() if len(node_parents) == 0]

    @property
    def non_roots(self) -> list[str]:
        return [node for node, node_parents in self.parents.items() if len(node_parents) > 0]

    @property
    def topo(self) -> list[str]:
        return topological_sort(self.parents)

    def children_map(self) -> dict[str, list[str]]:
        """Return ``node -> children``."""
        children: dict[str, list[str]] = {node: [] for node in self.nodes}
        for node, node_parents in self.parents.items():
            for parent in node_parents:
                children[parent].append(node)
        return children

    def parents_of(self, node: str) -> list[str]:
        """Return the parents of ``node``."""
        if node not in self.parents:
            raise KeyError(f"Unknown node '{node}'.")
        return list(self.parents[node])

    def validate_nodes(self, nodes: Iterable[str]) -> None:
        """Raise if any element of ``nodes`` is not in the DAG."""
        known = set(self.nodes)
        for node in nodes:
            if node not in known:
                raise ValueError(f"Unknown DAG node '{node}'.")

    def ancestors_of(self, nodes: Iterable[str], *, include_self: bool = True) -> tuple[str, ...]:
        """Return the ancestors of ``nodes`` in topological order."""
        selected = {str(node) for node in nodes}
        self.validate_nodes(selected)

        ancestors = set(selected) if include_self else set()
        stack = list(selected)
        while stack:
            node = stack.pop()
            for parent in self.parents_of(node):
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)

        return tuple(node for node in self.topo if node in ancestors)

    def moralized_graph(self, *, nodes: Iterable[str] | None = None) -> UndirectedAdjacency:
        """Return the moral graph using ``moralize_dag``."""
        return moralize_dag(self.parents, nodes=nodes)
