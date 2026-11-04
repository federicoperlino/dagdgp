"""Chordal graph construction utilities for DAG-SVI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import networkx as nx

from ..dag import DAG, UndirectedAdjacency


def _sorted_tuple(values: Iterable[str], *, order: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=lambda item: order[item]))


def _nx_graph_from_adjacency(adjacency: Mapping[str, Sequence[str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(adjacency.keys())
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            graph.add_edge(str(node), str(neighbor))
    return graph


def _adjacency_from_nx_graph(graph: nx.Graph, *, order: Mapping[str, int]) -> UndirectedAdjacency:
    return {node: set(_sorted_tuple(graph.neighbors(node), order=order)) for node in _sorted_tuple(graph.nodes, order=order)}


def is_perfect_elimination_order(adjacency: Mapping[str, Sequence[str]], order: Sequence[str]) -> bool:
    """Return whether ``order`` is a perfect elimination ordering."""
    position = {str(node): idx for idx, node in enumerate(order)}
    for idx, node in enumerate(order):
        later = [neighbor for neighbor in adjacency[str(node)] if position[neighbor] > idx]
        later_set = set(later)
        for left in later:
            if later_set - {left} - set(adjacency[left]):
                return False
    return True


def maximum_cardinality_search_order(
    adjacency: Mapping[str, Sequence[str]],
    *,
    reference_order: Sequence[str],
) -> tuple[str, ...]:
    """Return a deterministic perfect elimination ordering for a chordal graph."""
    nodes = tuple(str(node) for node in reference_order)
    node_set = set(nodes)
    if node_set != set(adjacency.keys()):
        raise ValueError("reference_order must contain exactly the graph nodes.")

    weights = {node: 0 for node in nodes}
    unnumbered = set(nodes)
    selected: list[str] = []
    reference_rank = {node: idx for idx, node in enumerate(nodes)}

    for _ in nodes:
        node = max(unnumbered, key=lambda item: (weights[item], -reference_rank[item]))
        unnumbered.remove(node)
        selected.append(node)
        for neighbor in adjacency[node]:
            if neighbor in unnumbered:
                weights[neighbor] += 1

    candidates = (tuple(reversed(selected)), tuple(selected))
    for candidate in candidates:
        if is_perfect_elimination_order(adjacency, candidate):
            return candidate
    raise ValueError("Failed to construct a perfect elimination ordering.")


def chordal_completion(
    adjacency: Mapping[str, Sequence[str]],
    *,
    reference_order: Sequence[str],
) -> UndirectedAdjacency:
    """Return a chordal completion of an undirected graph."""
    graph = _nx_graph_from_adjacency(adjacency)
    if not nx.is_chordal(graph):
        graph, _ = nx.complete_to_chordal_graph(graph)
    return _adjacency_from_nx_graph(graph, order={node: idx for idx, node in enumerate(reference_order)})


@dataclass(frozen=True)
class ChordalStructure:
    """Fixed sparsity pattern derived from the observed ancestor moral graph."""

    latent_nodes: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    ancestor_nodes: tuple[str, ...]
    order: tuple[str, ...]
    adjacency: Mapping[str, frozenset[str]]
    lower_block_pairs: tuple[tuple[int, int], ...]
    lower_children: tuple[tuple[int, ...], ...]
    maximal_cliques: tuple[tuple[str, ...], ...]

    @property
    def node_to_index(self) -> dict[str, int]:
        return {node: idx for idx, node in enumerate(self.order)}


def build_chordal_structure(
    dag: DAG,
    *,
    observed_nodes: Iterable[str],
) -> ChordalStructure:
    """Build the chordal sparsity pattern for the latent nodes of ``dag``."""
    latent_nodes = tuple(dag.non_roots)
    latent_set = set(latent_nodes)
    dag_order = {node: idx for idx, node in enumerate(dag.topo)}

    observed = _sorted_tuple((node for node in observed_nodes if node in latent_set), order=dag_order)
    unknown = sorted(set(observed_nodes) - set(dag.nodes))
    if unknown:
        raise ValueError(f"Unknown observed nodes: {unknown}.")

    if len(observed) == 0:
        adjacency = {node: frozenset() for node in latent_nodes}
        order = latent_nodes
        maximal_cliques = tuple((node,) for node in latent_nodes)
    else:
        ancestors = tuple(node for node in dag.ancestors_of(observed, include_self=True) if node in latent_set)
        moral = dag.moralized_graph(nodes=ancestors)
        chordal = chordal_completion(moral, reference_order=ancestors)
        full_adjacency = {node: set(chordal.get(node, set())) for node in latent_nodes}
        for node in latent_nodes:
            full_adjacency.setdefault(node, set())
        adjacency = {node: frozenset(_sorted_tuple(neighbors, order=dag_order)) for node, neighbors in full_adjacency.items()}

        chordal_graph = _nx_graph_from_adjacency(adjacency)
        order = maximum_cardinality_search_order(adjacency, reference_order=latent_nodes)
        clique_order = {node: idx for idx, node in enumerate(order)}
        maximal_cliques = tuple(
            sorted(
                (
                    _sorted_tuple(clique, order=clique_order)
                    for clique in nx.chordal_graph_cliques(chordal_graph)
                ),
                key=lambda clique: tuple(clique_order[node] for node in clique),
            )
        )

    node_to_index = {node: idx for idx, node in enumerate(order)}
    lower_pairs = tuple(
        (node_to_index[neighbor], node_to_index[node])
        for node in order
        for neighbor in sorted(adjacency[node], key=lambda item: node_to_index[item])
        if node_to_index[neighbor] > node_to_index[node]
    )
    lower_children = tuple(
        tuple(
            sorted(
                (node_to_index[neighbor] for neighbor in adjacency[node] if node_to_index[neighbor] > idx),
            )
        )
        for idx, node in enumerate(order)
    )
    ancestor_nodes = tuple(node for node in order if node in set(dag.ancestors_of(observed, include_self=True))) if observed else tuple()
    return ChordalStructure(
        latent_nodes=latent_nodes,
        observed_nodes=observed,
        ancestor_nodes=ancestor_nodes,
        order=order,
        adjacency={node: frozenset(adjacency[node]) for node in order},
        lower_block_pairs=lower_pairs,
        lower_children=lower_children,
        maximal_cliques=maximal_cliques,
    )


__all__ = [
    "ChordalStructure",
    "build_chordal_structure",
    "chordal_completion",
    "is_perfect_elimination_order",
    "maximum_cardinality_search_order",
]
