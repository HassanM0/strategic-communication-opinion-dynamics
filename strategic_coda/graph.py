"""
graph.py — Network construction for the strategic-CODA simulator.

Two public functions:
  build_base_graph              : undirected base topology + layout positions
  build_directed_weight_matrix  : communication digraph + row-stochastic
                                  influence matrix W = [w_ij]

Convention (paper, Sec. II): w_ij > 0 is the influence of source j on
receiver i; the sources of i are N_i^S = {j : w_ij > 0} and its audience is
N_i^A = {k : w_ki > 0}. W is row-stochastic with w_ii = 0.
"""

from __future__ import annotations

import functools

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Layout caching (Phase 2.3)
# Expensive layouts (spring, kamada_kawai) are cached by (graph_type, N, seed)
# so repeated widget interactions don't recompute them.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def _cached_spring_layout(graph_hash: int, seed: int) -> dict:
    """Return spring layout for a graph identified by its hash."""
    # This is called from build_base_graph; the graph is passed as hash only.
    # We store the graph alongside the cache key in _layout_graph_store.
    G = _layout_graph_store.get(graph_hash)
    if G is None:
        return {}
    return nx.spring_layout(G, seed=seed)


@functools.lru_cache(maxsize=128)
def _cached_kamada_kawai_layout(graph_hash: int) -> dict:
    G = _layout_graph_store.get(graph_hash)
    if G is None:
        return {}
    return nx.kamada_kawai_layout(G)


@functools.lru_cache(maxsize=128)
def _cached_spectral_layout(graph_hash: int) -> dict:
    G = _layout_graph_store.get(graph_hash)
    if G is None:
        return {}
    return nx.spectral_layout(G)


# Simple graph-object store keyed by id so lru_cache can work on hashable keys.
_layout_graph_store: dict[int, nx.Graph] = {}


def _get_layout(G: nx.Graph, algo: str, seed: int) -> dict:
    """Compute (and cache where possible) a node-position dict for graph G.

    Parameters
    ----------
    G : nx.Graph
        Undirected base graph.
    algo : str
        Layout algorithm name: 'spring', 'circular', 'shell', 'spectral',
        'kamada_kawai'.
    seed : int
        RNG seed (used for spring layout).

    Returns
    -------
    dict
        Mapping node → (x, y) position.
    """
    gh = id(G)
    _layout_graph_store[gh] = G

    if algo == "spring":
        return _cached_spring_layout(gh, seed)
    elif algo == "circular":
        return nx.circular_layout(G)
    elif algo == "shell":
        N = G.number_of_nodes()
        if N > 1:
            return nx.shell_layout(G, nlist=[[0], list(range(1, N))])
        return nx.circular_layout(G)
    elif algo == "spectral":
        return _cached_spectral_layout(gh)
    elif algo == "kamada_kawai":
        return _cached_kamada_kawai_layout(gh)
    else:
        return nx.spring_layout(G, seed=seed)


# ---------------------------------------------------------------------------
# Base graph builders
# ---------------------------------------------------------------------------

def build_base_graph(
    num_agents: int,
    graph_type: str = "watts_strogatz",
    seed: int = 12,
    regular_degree: int = 3,
    layout_algo: str = "spring",
    sbm_sizes: str = "10,10",
    sbm_p_in: float = 0.35,
    sbm_p_out: float = 0.06,
    ws_k: int = 4,
    ws_p: float = 0.1,
    group_sizes: str = "2,3",
) -> tuple[nx.Graph, dict]:
    """Build an undirected base graph and compute its layout positions.

    Parameters
    ----------
    num_agents : int
        Number of nodes N.
    graph_type : str
        Topology identifier.  Supported values:
        'watts_strogatz', 'complete', 'star', 'wheel', 'line', 'grid', 'ring',
        'stochastic_block', 'd_regular', 'bipartite', 'powerlaw_cluster',
        'erdos_renyi', 'karate_club' (N=34), 'petersen' (N=10),
        'dodecahedral' (N=20), 'two_group' (complete graph whose nodes are
        split into consecutive groups of sizes ``group_sizes``; node
        attribute 'group').
    seed : int
        Random seed for stochastic generators.
    regular_degree : int
        Degree used for 'd_regular' topology.
    layout_algo : str
        Visualisation layout: 'spring', 'circular', 'shell', 'spectral',
        'kamada_kawai'.

    Returns
    -------
    G : nx.Graph
        Undirected base graph with integer node labels 0 … N-1.
    pos : dict
        Node → (x, y) position for visualisation.

    Raises
    ------
    ValueError
        If parameters are invalid or incompatible.
    """
    N = int(num_agents)
    seed = int(seed)

    if N < 2:
        raise ValueError("Number of agents must be at least 2.")

    if graph_type == "watts_strogatz":
        k_ws = min(ws_k, N - 1)
        if k_ws < 2:
            G = nx.complete_graph(n=N)
        else:
            G = nx.watts_strogatz_graph(n=N, k=k_ws, p=ws_p, seed=seed)

    elif graph_type == "complete":
        G = nx.complete_graph(n=N)

    elif graph_type == "star":
        G = nx.star_graph(n=N - 1)

    elif graph_type == "wheel":
        if N < 4:
            raise ValueError("Wheel graph requires at least 4 agents.")
        G = nx.wheel_graph(n=N)

    elif graph_type == "line":
        G = nx.path_graph(n=N)

    elif graph_type == "ring":
        G = nx.cycle_graph(n=N)

    elif graph_type == "grid":
        rows = max(int(np.floor(np.sqrt(N))), 1)
        cols = int(np.ceil(N / rows))
        grid_2d = nx.grid_2d_graph(rows, cols)
        mapping = {node: idx for idx, node in enumerate(grid_2d.nodes())}
        grid_int = nx.relabel_nodes(grid_2d, mapping)
        G = grid_int.subgraph(range(N)).copy()

    elif graph_type == "stochastic_block":
        try:
            sizes = [int(s.strip()) for s in sbm_sizes.split(",")]
        except ValueError:
            sizes = [N // 2, N - N // 2]
        
        # Adjust sizes to match total N if they don't match exactly.
        # Prioritise user's proportions but ensure sum(sizes) == N
        total_size = sum(sizes)
        if total_size != N and total_size > 0:
             sizes = [max(1, int(s / total_size * N)) for s in sizes]
             # fix round-off
             diff = N - sum(sizes)
             sizes[-1] += diff # put remainder in the last block
        elif total_size == 0 or any(s <= 0 for s in sizes):
             sizes = [N // 2, N - N // 2]
             
        k = len(sizes)
        probs = np.full((k, k), sbm_p_out)
        np.fill_diagonal(probs, sbm_p_in)
        G = nx.stochastic_block_model(sizes=sizes, p=probs, seed=seed)

    elif graph_type == "d_regular":
        d = int(np.clip(regular_degree, 0, N - 1))
        if d == 0:
            G = nx.empty_graph(n=N)
        else:
            if (N * d) % 2 != 0:
                raise ValueError(
                    f"For d-regular graphs, N*d must be even (N={N}, d={d})."
                )
            G = nx.random_regular_graph(d=d, n=N, seed=seed)

    elif graph_type == "bipartite":
        n1 = N // 2
        n2 = N - n1
        rng = np.random.default_rng(seed)
        G = nx.bipartite.random_graph(n=n1, m=n2, p=0.4, seed=int(rng.integers(1e6)))
        # Relabel to integer 0…N-1
        G = nx.convert_node_labels_to_integers(G)

    elif graph_type == "powerlaw_cluster":
        m = max(1, min(regular_degree, N - 1))
        G = nx.powerlaw_cluster_graph(n=N, m=m, p=0.1, seed=seed)

    elif graph_type == "erdos_renyi":
        # ensure connected
        p_val = 0.3
        G = nx.erdos_renyi_graph(n=N, p=p_val, seed=seed)
        attempts = 0
        while not nx.is_connected(G) and attempts < 100:
            seed += 1
            G = nx.erdos_renyi_graph(n=N, p=p_val, seed=seed)
            attempts += 1
            
    elif graph_type == "karate_club":
        if N != 34:
            raise ValueError("Karate club graph strictly requires N=34 agents.")
        G = nx.karate_club_graph()

    elif graph_type == "petersen":
        if N != 10:
            raise ValueError("Petersen graph strictly requires N=10 agents.")
        G = nx.petersen_graph()

    elif graph_type == "dodecahedral":
        if N != 20:
            raise ValueError("Dodecahedral graph strictly requires N=20 agents.")
        G = nx.dodecahedral_graph()

    elif graph_type == "two_group":
        try:
            sizes = [int(s.strip()) for s in str(group_sizes).split(",") if s.strip()]
        except ValueError:
            raise ValueError(f"group_sizes must be comma-separated integers, got {group_sizes!r}")
        if len(sizes) != 2 or min(sizes) < 1 or sum(sizes) != N:
            raise ValueError(
                f"two_group needs two positive group sizes summing to N={N}; "
                f"got {group_sizes!r}.")
        G = nx.complete_graph(n=N)
        for node in G.nodes():
            G.nodes[node]["group"] = 0 if node < sizes[0] else 1

    else:
        raise ValueError(
            f"Unknown graph_type '{graph_type}'. Supported: "
            "watts_strogatz, complete, star, wheel, line, grid, ring, stochastic_block, "
            "d_regular, bipartite, powerlaw_cluster, erdos_renyi, karate_club, "
            "petersen, dodecahedral, two_group."
        )

    # Ensure node labels are integers 0…N-1
    G = nx.convert_node_labels_to_integers(G)

    # Fix any arbitrarily isolated nodes to prevent 0-audience NaN calculation issues
    # This acts as a universal safety net for sparse random graphs (e.g. SBM, ER)
    isolated = list(nx.isolates(G))
    if isolated:
        rng = np.random.default_rng(seed)
        for node in isolated:
            target = int(rng.integers(0, N))
            while target == node:
                target = int(rng.integers(0, N))
            G.add_edge(node, target)

    # Compute layout ---------------------------------------------------------
    if graph_type == "grid":
        rows_g = max(int(np.floor(np.sqrt(N))), 1)
        cols_g = int(np.ceil(N / rows_g))
        grid_2d_ref = nx.grid_2d_graph(rows_g, cols_g)
        mapping_ref = {node: idx for idx, node in enumerate(grid_2d_ref.nodes())}
        int_pos = {idx: (node[1], -node[0]) for node, idx in mapping_ref.items()}
        pos = {i: int_pos[i] for i in G.nodes() if i in int_pos}
    else:
        pos = _get_layout(G, layout_algo, seed)

    return G, pos


# ---------------------------------------------------------------------------
# Directed weight matrix
# ---------------------------------------------------------------------------

def build_directed_weight_matrix(
    num_agents: int,
    graph_type: str = "watts_strogatz",
    seed: int = 12,
    regular_degree: int = 3,
    directed: bool = True,
    directed_mode: str = "strongly_connected",
    weight_mode: str = "uniform_in",
    ensure_valid_graph: bool = True,
    weight_seed: int = 97,
    layout_algo: str = "spring",
    sbm_sizes: str = "10,10",
    sbm_p_in: float = 0.35,
    sbm_p_out: float = 0.06,
    ws_k: int = 4,
    ws_p: float = 0.1,
    group_sizes: str = "2,3",
    cross_group_weight: float = 0.2,
) -> dict:
    """Build the communication digraph and the row-stochastic matrix W.

    ``W[i, j] = w_ij > 0`` means receiver i places weight on source j.
    Row sums satisfy Σ_j w_ij = 1 for every agent with at least one source,
    and w_ii = 0 (no self-loops).

    Parameters
    ----------
    num_agents : int
        Number of agents N.
    graph_type : str
        Base topology (see build_base_graph).
    seed : int
        Graph generation seed.
    regular_degree : int
        Degree for d_regular topology.
    directed : bool
        If False, every base edge is represented in both directions. If True,
        orient the base topology according to ``directed_mode``.
    directed_mode : str
        ``strongly_connected`` keeps a seeded spanning tree reciprocal and
        orients every non-tree edge one way. ``random_asymmetric`` randomly
        orients edges and repairs zero in/out-degree using reverse arcs on the
        same base edges. ``line_directed`` and ``ring_directed`` are available
        only for their matching base topologies.
    weight_mode : str
        'uniform_in'     — equal weights over the sources of each receiver.
        'random_in'      — weights drawn uniformly from (0.1, 1), normalised.
        'out_degree' / 'in_degree' / 'eigenvector'
                         — proportional to the source's degree / centrality.
        'karate_weights' — proportional to the 'weight' edge attribute
                           (interaction counts of the Karate Club data set).
        'group_weights'  — 'two_group' network: same-group sources share
                           1 - cross_group_weight uniformly, other-group
                           sources share cross_group_weight uniformly.
    ensure_valid_graph : bool
        When True, raise ValueError if the graph is disconnected or has
        isolated nodes.
    weight_seed : int
        RNG seed for random weight sampling.
    layout_algo : str
        Layout algorithm passed through to build_base_graph.

    Returns
    -------
    dict with keys:
        'graph'       : nx.DiGraph
        'base_graph'  : nx.Graph  (undirected)
        'layout_pos'  : dict (node → position)
        'W'           : np.ndarray shape (N, N), W[i,j] = weight i on j
        'in_degree'   : np.ndarray shape (N,) int
        'out_degree'  : np.ndarray shape (N,) int
    """
    N = int(num_agents)

    base_graph, layout_pos = build_base_graph(
        num_agents=N,
        graph_type=graph_type,
        seed=seed,
        regular_degree=regular_degree,
        layout_algo=layout_algo,
        sbm_sizes=sbm_sizes,
        sbm_p_in=sbm_p_in,
        sbm_p_out=sbm_p_out,
        ws_k=ws_k,
        ws_p=ws_p,
        group_sizes=group_sizes,
    )

    # Build the communication digraph while preserving the base skeleton.
    # An arc (u, v) means that u is a source observed by receiver v.
    G = nx.DiGraph()
    G.add_nodes_from(base_graph.nodes())
    edges = sorted((min(int(u), int(v)), max(int(u), int(v)))
                   for u, v in base_graph.edges())
    rng_dir = np.random.default_rng(int(seed) + 42)

    if not directed:
        for u, v in edges:
            G.add_edge(u, v)
            G.add_edge(v, u)

    elif directed_mode == "strongly_connected":
        if not nx.is_connected(base_graph):
            raise ValueError(
                "Strong-preserving direction requires a connected base graph. "
                "Change the graph seed/parameters or use Random asymmetric."
            )

        # A reciprocal spanning tree alone is strongly connected. Synthetic,
        # seeded priorities choose the tree without mutating scientific edge
        # attributes such as the Karate-club interaction weights.
        prioritized = nx.Graph()
        prioritized.add_nodes_from(base_graph.nodes())
        for u, v in edges:
            prioritized.add_edge(u, v, _orientation_priority=float(rng_dir.random()))
        spanning_tree = nx.minimum_spanning_tree(
            prioritized, weight="_orientation_priority", algorithm="kruskal")
        tree_edges = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in spanning_tree.edges()
        }

        for u, v in edges:
            if (u, v) in tree_edges:
                G.add_edge(u, v)
                G.add_edge(v, u)
            elif rng_dir.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    elif directed_mode == "random_asymmetric":
        for u, v in edges:
            if rng_dir.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

        # Keep every agent a receiver and a sender without inventing social
        # ties: repairs only add the reverse arc of an existing base edge.
        for i in range(N):
            neighbors = sorted(int(n) for n in base_graph.neighbors(i))
            if neighbors and G.in_degree(i) == 0:
                source = neighbors[int(rng_dir.integers(len(neighbors)))]
                G.add_edge(source, i)
            if neighbors and G.out_degree(i) == 0:
                target = neighbors[int(rng_dir.integers(len(neighbors)))]
                G.add_edge(i, target)

    elif directed_mode == "line_directed":
        if graph_type != "line":
            raise ValueError("Directed Path is available only for the Line graph.")
        for i in range(N - 1):
            G.add_edge(i, i + 1)
        # Endpoint-safe reciprocal links keep W row-stochastic and xi finite.
        if N >= 2:
            G.add_edge(1, 0)
            G.add_edge(N - 1, N - 2)

    elif directed_mode == "ring_directed":
        if graph_type != "ring":
            raise ValueError("Directed Cycle is available only for the Ring graph.")
        for i in range(N):
            G.add_edge(i, (i + 1) % N)

    else:
        raise ValueError(
            "directed_mode must be one of {'strongly_connected', "
            "'random_asymmetric', 'line_directed', 'ring_directed'}."
        )

    is_weakly_connected = bool(nx.is_weakly_connected(G))
    is_strongly_connected = bool(nx.is_strongly_connected(G))
    strongly_connected_components = list(nx.strongly_connected_components(G))
    scc_count = len(strongly_connected_components)
    largest_scc_size = max((len(c) for c in strongly_connected_components), default=0)
    has_one_way_edges = any(not G.has_edge(v, u) for u, v in G.edges())

    # Build weight matrix W[i, j]: receiver i, source j
    W = np.zeros((N, N), dtype=float)
    rng = np.random.default_rng(int(weight_seed))

    eigenvector_centrality = None

    for i in range(N):
        sources = np.array(list(G.predecessors(i)), dtype=int)
        if sources.size == 0:
            continue
        if weight_mode == "uniform_in":
            incoming = np.full(sources.size, 1.0 / sources.size)
        elif weight_mode == "random_in":
            raw = rng.uniform(0.1, 1.0, size=sources.size)
            incoming = raw / raw.sum()
        elif weight_mode == "out_degree":
            raw = np.array([G.out_degree(src) for src in sources], dtype=float)
            incoming = raw / raw.sum() if raw.sum() > 0 else np.full(sources.size, 1.0 / sources.size)
        elif weight_mode == "in_degree":
            raw = np.array([G.in_degree(src) for src in sources], dtype=float)
            incoming = raw / raw.sum() if raw.sum() > 0 else np.full(sources.size, 1.0 / sources.size)
        elif weight_mode == "eigenvector":
            if eigenvector_centrality is None:
                try:
                    eigenvector_centrality = nx.eigenvector_centrality_numpy(base_graph)
                except Exception:
                    eigenvector_centrality = {n: 1.0 for n in base_graph.nodes()}
            raw = np.array([eigenvector_centrality[src] for src in sources], dtype=float)
            incoming = raw / raw.sum() if raw.sum() > 0 else np.full(sources.size, 1.0 / sources.size)
        elif weight_mode == "karate_weights":
            raw = np.array([base_graph[src][i].get("weight", 1.0) for src in sources], dtype=float)
            incoming = raw / raw.sum() if raw.sum() > 0 else np.full(sources.size, 1.0 / sources.size)
        elif weight_mode == "group_weights":
            groups = nx.get_node_attributes(base_graph, "group")
            if len(groups) != N:
                raise ValueError("weight_mode='group_weights' requires graph_type='two_group'.")
            rho = float(cross_group_weight)
            if not 0.0 <= rho <= 1.0:
                raise ValueError("cross_group_weight must lie in [0, 1].")
            same = np.array([groups[int(src)] == groups[i] for src in sources])
            incoming = np.zeros(sources.size, dtype=float)
            if not same.any():
                incoming[:] = 1.0 / sources.size
            elif same.all():
                incoming[:] = 1.0 / sources.size
            else:
                incoming[same] = (1.0 - rho) / int(same.sum())
                incoming[~same] = rho / int((~same).sum())
        else:
            raise ValueError(
                "weight_mode must be 'uniform_in', 'random_in', 'out_degree', "
                "'in_degree', 'eigenvector', 'karate_weights', or 'group_weights'.")
        W[i, sources] = incoming

    in_degree = np.array([G.in_degree(i) for i in range(N)], dtype=int)
    out_degree = np.array([G.out_degree(i) for i in range(N)], dtype=int)

    if ensure_valid_graph:
        if base_graph.number_of_edges() == 0:
            raise ValueError(
                "Graph has no edges. Choose a connected topology or "
                "disable strict validation."
            )
        if directed:
            if not is_strongly_connected:
                raise ValueError(
                    "Generated directed network is not strongly connected "
                    f"({scc_count} strongly connected components; largest has "
                    f"{largest_scc_size}/{N} agents)."
                )
        elif not nx.is_connected(base_graph):
            raise ValueError(
                "Generated topology is not connected. "
                "Change seed or graph parameters."
            )
        if np.any(in_degree == 0):
            bad = np.where(in_degree == 0)[0].tolist()
            raise ValueError(
                f"Invalid network: receivers with no sources: {bad[:8]}"
            )
        if np.any(out_degree == 0):
            bad = np.where(out_degree == 0)[0].tolist()
            raise ValueError(
                f"Invalid network: senders with no audience: {bad[:8]}"
            )
        active = in_degree > 0
        row_sums = W.sum(axis=1)
        if not np.allclose(row_sums[active], 1.0, atol=1e-10):
            raise ValueError("Incoming weights are not properly normalised per receiver.")

    return {
        "graph": G,
        "base_graph": base_graph,
        "layout_pos": layout_pos,
        "W": W,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "is_weakly_connected": is_weakly_connected,
        "is_strongly_connected": is_strongly_connected,
        "scc_count": scc_count,
        "largest_scc_size": largest_scc_size,
        "has_one_way_edges": has_one_way_edges,
    }
