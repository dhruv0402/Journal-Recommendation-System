STRONG_MATCH_THRESHOLD = 0.75
WEAK_MATCH_THRESHOLD = 0.45

# Domain keyword → reason template mapping
# Used to generate dynamic, abstract-grounded reasons instead of hardcoded text
DOMAIN_SIGNALS = [
    (
        [
            "neural network",
            "deep learning",
            "cnn",
            "transformer",
            "optimizer",
            "gradient",
            "backpropagation",
            "training",
            "classification",
            "image",
        ],
        "AI/ML methodology",
        "machine learning models and data-driven techniques",
    ),
    (
        [
            "routing",
            "network",
            "protocol",
            "wireless",
            "topology",
            "bandwidth",
            "latency",
            "packet",
            "mesh",
            "sdn",
            "tcp",
            "udp",
            "congestion",
        ],
        "network systems",
        "network systems, routing, and distributed communication",
    ),
    (
        [
            "federated",
            "intrusion",
            "security",
            "encryption",
            "cryptograph",
            "attack",
            "malware",
            "firewall",
            "authentication",
            "privacy",
        ],
        "cybersecurity",
        "security systems, threat detection, and cryptographic methods",
    ),
    (
        [
            "parallel",
            "cuda",
            "gpu",
            "hpc",
            "cache",
            "numa",
            "multicore",
            "thread",
            "distributed computing",
            "mpi",
            "openmp",
            "sparse matrix",
        ],
        "parallel computing",
        "parallel and high-performance computing systems",
    ),
    (
        [
            "software",
            "testing",
            "refactor",
            "agile",
            "devops",
            "debugging",
            "code quality",
            "maintenance",
            "architecture",
            "uml",
            "design pattern",
        ],
        "software engineering",
        "software development, testing, and engineering practices",
    ),
    (
        [
            "image",
            "vision",
            "object detection",
            "segmentation",
            "recognition",
            "feature extraction",
            "convolutional",
            "pixel",
            "visual",
        ],
        "computer vision",
        "image processing, visual recognition, and computer vision",
    ),
    (
        [
            "graph",
            "node",
            "edge",
            "spectral",
            "eigenvalue",
            "adjacency",
            "random graph",
            "network analysis",
            "centrality",
        ],
        "graph theory",
        "graph structures, network analysis, and combinatorial methods",
    ),
    (
        [
            "reinforcement learning",
            "reward",
            "policy",
            "agent",
            "mdp",
            "q-learning",
            "actor critic",
            "environment",
            "exploration",
        ],
        "reinforcement learning",
        "reinforcement learning and autonomous decision-making systems",
    ),
    (
        [
            "iot",
            "sensor",
            "embedded",
            "edge computing",
            "actuator",
            "real-time",
            "microcontroller",
            "smart",
            "monitoring",
        ],
        "IoT and embedded systems",
        "IoT systems, sensor networks, and embedded computing",
    ),
    (
        [
            "database",
            "sql",
            "query",
            "indexing",
            "transaction",
            "nosql",
            "data warehouse",
            "etl",
            "schema",
            "relational",
        ],
        "database systems",
        "database design, query optimization, and data management",
    ),
]

# Per-journal scope descriptions for dynamic reason generation
JOURNAL_SCOPE = {
    "Computer Networks": (
        "specializes in network-oriented research",
        "strong",
    ),
    "Computer Communications": (
        "specializes in network-oriented research",
        "strong",
    ),
    "Engineering Applications of Artificial Intelligence": (
        "emphasizes AI methodologies applied to engineering problems",
        "suitable",
    ),
    "Artificial Intelligence": (
        "focuses on core AI research and intelligent systems",
        "suitable",
    ),
    "AI Open": (
        "covers open AI research and machine learning advances",
        "suitable",
    ),
    "Parallel Computing": (
        "focuses on parallel and distributed computing systems",
        "suitable",
    ),
    "Journal of Systems and Software": (
        "covers software systems design and engineering",
        "moderate",
    ),
    "Advances in Engineering Software": (
        "publishes on software tools for engineering applications",
        "moderate",
    ),
    "Computer Vision and Image Understanding": (
        "specializes in visual computing and image analysis",
        "strong",
    ),
    "Computer Standards & Interfaces": (
        "covers software standards, protocols, and interface design",
        "moderate",
    ),
    "Journal of Computer and System Sciences": (
        "covers theoretical computer science and system design",
        "moderate",
    ),
}


def _detect_domain(abstract: str) -> tuple:
    """
    Detect the research domain from abstract keywords.
    Returns (domain_label, domain_description).
    """
    if not abstract:
        return "general computational methods", "general computational methods"

    text = abstract.lower()
    best_domain = ("general computational methods", "general computational methods")
    best_count = 0

    for keywords, label, description in DOMAIN_SIGNALS:
        count = sum(1 for kw in keywords if kw in text)
        if count > best_count:
            best_count = count
            best_domain = (label, description)

    return best_domain


def _generate_reason(journal_name: str, similarity: float, abstract: str) -> str:
    """
    Generate a dynamic, abstract-grounded reason for a journal recommendation.
    No more hardcoded 'network systems, routing' for every journal.
    """
    domain_label, domain_desc = _detect_domain(abstract)
    scope_desc, fit_word = JOURNAL_SCOPE.get(
        journal_name,
        (f"covers research in {domain_desc}", "possible"),
    )

    if similarity >= 0.65:
        return (
            f"{journal_name} {scope_desc}, and the paper's focus on "
            f"{domain_desc} aligns well with its scope."
        )
    elif similarity >= 0.45:
        return (
            f"{journal_name} has moderate overlap with {domain_desc}, "
            f"making it a possible but not perfect match."
        )
    else:
        return (
            f"{journal_name} has limited alignment with {domain_desc}, "
            f"suggesting it may not be the best venue."
        )


def clamp(score: float) -> float:
    return max(0.0, min(1.0, round(float(score), 3)))


def make_final_decision(
    journal_predictions, semantic_validation=None, abstract: str = ""
):
    """
    Returns structured dict:
    {
        "journal": str,
        "confidence": float,
        "reason": str   ← now dynamic, grounded in abstract content
    }
    """
    if not journal_predictions:
        return {
            "journal": "No suitable journal",
            "confidence": 0.0,
            "reason": "No matching journals found in dataset.",
        }

    top = journal_predictions[0]
    journal_name = top.get("journal_name", "Unknown")
    top_confidence = float(top.get("confidence", 0.0))
    top_similarity = float(top.get("similarity", 0.0))

    # ---------- SEMANTIC OVERRIDE ----------
    if semantic_validation:
        topic_alignment = clamp(semantic_validation.get("topic_alignment", 0.0))
        embedding_similarity = clamp(
            semantic_validation.get("embedding_similarity", 0.0)
        )
        techniques = semantic_validation.get("techniques", [])
        strong_technique = any(t.get("confidence", 0) >= 0.85 for t in techniques)

        if topic_alignment >= 0.8 and embedding_similarity < 0.6 and strong_technique:
            return {
                "journal": journal_name,
                "confidence": max(top_confidence, 0.65),
                "reason": _generate_reason(journal_name, top_similarity, abstract),
            }

    # ---------- NORMAL DECISION ----------
    # Also inject dynamic reason into all journal predictions
    for j in journal_predictions:
        j["reason"] = _generate_reason(
            j.get("journal_name", ""),
            j.get("similarity", 0.0),
            abstract,
        )

    return {
        "journal": journal_name,
        "confidence": top_confidence,
        "reason": _generate_reason(journal_name, top_similarity, abstract),
    }
