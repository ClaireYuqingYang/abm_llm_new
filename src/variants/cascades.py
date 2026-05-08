"""Cascade-tree reconstruction and Vosoughi-style structural metrics.

A cascade is a rooted tree of nodes that *shared* a story.  We reconstruct
it from per-node parent pointers (the agent who first exposed the
sharer to the story).  Per-cascade we report size, depth, breadth, and
structural virality (Goel et al. 2016).
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


def compute_cascades(
    story: str,
    parent: np.ndarray,
    seed_arr: np.ndarray,
    shared: np.ndarray,
) -> list[dict]:
    """
    Compute per-seed cascade metrics from the parent pointers.

    Only nodes that *shared* the content count as cascade members — passive
    viewers do not extend the cascade tree.
    """
    n = len(parent)
    out: list[dict] = []

    # Build children adjacency restricted to nodes that shared.
    children: dict[int, list[int]] = defaultdict(list)
    for v in range(n):
        if not shared[v]:
            continue
        p = int(parent[v])
        if p == -1:
            continue  # seed
        if not shared[p]:
            continue
        children[p].append(v)

    seeds = [v for v in range(n) if shared[v] and parent[v] == -1]
    for seed in seeds:
        depths = {seed: 0}
        stack = [seed]
        order = [seed]
        while stack:
            node = stack.pop()
            for child in children.get(node, []):
                if child not in depths:
                    depths[child] = depths[node] + 1
                    stack.append(child)
                    order.append(child)
        size = len(order)
        depth = max(depths.values()) if depths else 0
        breadth = max(Counter(depths.values()).values()) if depths else 0
        sv = _structural_virality(order, children) if size >= 2 else 0.0

        out.append(
            {
                "story": story,
                "seed": int(seed),
                "size": int(size),
                "depth": int(depth),
                "breadth": int(breadth),
                "structural_virality": float(sv),
            }
        )
    return out


def _structural_virality(nodes: list[int], children: dict[int, list[int]]) -> float:
    """
    Average pairwise shortest-path distance in the cascade tree
    (Goel et al. 2016 definition).  For tree T with n nodes,
    SV(T) = (1 / n(n-1)) Σ_{i!=j} dist(i, j).
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    adj: dict[int, list[int]] = defaultdict(list)
    node_set = set(nodes)
    for parent, kids in children.items():
        if parent not in node_set:
            continue
        for k in kids:
            if k in node_set:
                adj[parent].append(k)
                adj[k].append(parent)

    total = 0.0
    pairs = 0
    for src in nodes:
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for tgt, d in dist.items():
            if tgt == src:
                continue
            total += d
            pairs += 1
    if pairs == 0:
        return 0.0
    return total / pairs
