import networkx as nx

# The five fault types our system can diagnose.
# These become the GOAL NODES in A* search.
FAULT_TYPES = [
    "Tool Wear Failure",
    "Heat Dissipation Failure",
    "Power Failure",
    "Overstrain Failure",
    "Random Failure",
    "No Failure",
]

# Signal nodes — these are the STARTING NODES fed in from NLP extraction.
SIGNAL_NODES = [
    "high_torque",
    "low_rpm",
    "high_process_temp",
    "high_air_temp",
    "wear_warning",
    "wear_critical",
    "normal_operation",
]

# Each tuple: (signal_node, fault_node, weight)
# Weight = how strongly this signal points to this fault.
# Higher weight = stronger evidence. Range: 0.0 to 1.0.
# In A* terms: LOWER edge cost = STRONGER evidence.
# So we store cost = 1.0 - weight internally.
EDGES = [
    # --- Tool Wear Failure ---
    # Caused by tools running past their service life
    ("wear_critical",    "Tool Wear Failure",         0.95),
    ("wear_warning",     "Tool Wear Failure",         0.60),
    ("high_torque",      "Tool Wear Failure",         0.40),

    # --- Heat Dissipation Failure ---
    # Caused by poor cooling, high ambient/process temperature
    ("high_process_temp","Heat Dissipation Failure",  0.90),
    ("high_air_temp",    "Heat Dissipation Failure",  0.70),
    ("low_rpm",          "Heat Dissipation Failure",  0.35),

    # --- Power Failure ---
    # Caused by extreme torque-speed combinations drawing excess power
    ("low_rpm",          "Power Failure",             0.65),
    ("high_torque",      "Power Failure",             0.55),
    ("wear_warning",     "Power Failure",             0.30),

    # --- Overstrain Failure ---
    # Caused by mechanical overload — high torque is primary indicator
    ("high_torque",      "Overstrain Failure",        0.85),
    ("wear_critical",    "Overstrain Failure",        0.30),
    ("low_rpm",          "Overstrain Failure",        0.40),

    # --- Random Failure ---
    # No clear pattern — weak connections from multiple signals
    ("high_air_temp",    "Random Failure",            0.20),
    ("wear_warning",     "Random Failure",            0.15),
    ("high_process_temp","Random Failure",            0.15),

    # --- No Failure ---
    # Normal operation signal points here
    ("normal_operation", "No Failure",                1.00),
]


def build_graph():
    # Creates a directed weighted graph.
    # Nodes: signal nodes + fault nodes
    # Edges: signal → fault with cost = 1.0 - weight
    # Lower cost = stronger evidence path for A* to prefer.

    G = nx.DiGraph()
    G.add_nodes_from(SIGNAL_NODES, node_type="signal")
    G.add_nodes_from(FAULT_TYPES, node_type="fault")

    for signal, fault, weight in EDGES:
        cost = round(1.0 - weight, 3)
        G.add_edge(signal, fault, weight=cost, evidence_strength=weight)

    return G


def get_active_signals(signals_dict):
    # Filters the NLP output to only the signals that fired True.
    # Returns a list of signal node names to use as A* start points.
    return [s for s, active in signals_dict.items() if active]


def score_faults(G, active_signals):
    # For each fault type, accumulate evidence from all active signals.
    # Score = sum of evidence_strength for all edges from active signals to that fault.
    # This gives us a ranked list of most likely faults.

    scores = {fault: 0.0 for fault in FAULT_TYPES}

    for signal in active_signals:
        if signal not in G:
            continue
        for neighbor in G.neighbors(signal):
            strength = G[signal][neighbor]["evidence_strength"]
            scores[neighbor] += strength

    return scores


def astar_diagnosis(G, active_signals):
    # Runs the diagnosis using A* search logic.
    #
    # In a standard A* over a graph: f(n) = g(n) + h(n)
    # Here:
    #   g(n) = accumulated cost along the path (lower = stronger evidence)
    #   h(n) = heuristic — we use 0 (admissible, so A* is optimal)
    #
    # We find the fault node reachable from the most active signals
    # with the lowest total cost path = highest total evidence.
    #
    # If multiple signals are active, we pick the fault with
    # the best combined evidence score across all signals.

    if not active_signals:
        return "No Failure", 0.0, []

    scores = score_faults(G, active_signals)

    # Find best fault (highest score)
    best_fault = max(scores, key=scores.get)
    best_score = scores[best_fault]

    if best_score == 0.0:
        return "No Failure", 0.0, []

    # Collect the contributing signals (evidence trail)
    evidence = []
    for signal in active_signals:
        if signal in G and G.has_edge(signal, best_fault):
            strength = G[signal][best_fault]["evidence_strength"]
            evidence.append((signal, strength))

    # Sort evidence by strength descending
    evidence.sort(key=lambda x: x[1], reverse=True)

    # Confidence: normalize score against maximum possible for this fault
    max_possible = sum(
        G[s][best_fault]["evidence_strength"]
        for s in SIGNAL_NODES
        if G.has_edge(s, best_fault)
    )
    confidence = round((best_score / max_possible) * 100, 1) if max_possible > 0 else 0.0

    return best_fault, confidence, evidence


if __name__ == "__main__":
    G = build_graph()

    print("=== GRAPH STRUCTURE ===")
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")

    print("\n=== A* DIAGNOSIS TESTS ===\n")

    test_cases = [
        {
            "name": "Normal operation",
            "signals": {"normal_operation": True, "high_torque": False,
                        "low_rpm": False, "high_process_temp": False,
                        "high_air_temp": False, "wear_warning": False, "wear_critical": False}
        },
        {
            "name": "Tool wear failure signals",
            "signals": {"wear_critical": True, "high_torque": True,
                        "normal_operation": False, "low_rpm": False,
                        "high_process_temp": False, "high_air_temp": False, "wear_warning": False}
        },
        {
            "name": "Heat dissipation signals",
            "signals": {"high_process_temp": True, "high_air_temp": True,
                        "low_rpm": True, "normal_operation": False,
                        "high_torque": False, "wear_warning": False, "wear_critical": False}
        },
        {
            "name": "Power failure signals",
            "signals": {"low_rpm": True, "high_torque": True,
                        "wear_warning": True, "normal_operation": False,
                        "high_process_temp": False, "high_air_temp": False, "wear_critical": False}
        },
        {
            "name": "Overstrain signals",
            "signals": {"high_torque": True, "low_rpm": True,
                        "wear_critical": True, "normal_operation": False,
                        "high_process_temp": False, "high_air_temp": False, "wear_warning": False}
        },
    ]

    for case in test_cases:
        active = get_active_signals(case["signals"])
        fault, confidence, evidence = astar_diagnosis(G, active)

        print(f"Test         : {case['name']}")
        print(f"Active signals: {active}")
        print(f"Diagnosis    : {fault}")
        print(f"Confidence   : {confidence}%")
        print(f"Evidence     : {[(s, round(w,2)) for s, w in evidence]}")
        print("-" * 70)