from typing import Dict, List

# -----------------------------
# Domain → Academic Title Map
# -----------------------------
DOMAIN_TITLES = {
    "networking": "Networked Systems",
    "biology": "Biological Interaction Networks",
    "security": "Secure and Trusted Systems",
    "distributed systems": "Distributed and Decentralized Systems",
    "wireless": "Wireless and Mobile Systems",
    "unknown": "Interdisciplinary Systems"
}

# -----------------------------
# Technique → Phrase Map
# -----------------------------
TECHNIQUE_MAP = {
    "spanner": "Graph Spanner Theory",
    "graph": "Graph-Theoretic Methods",
    "optimization": "Optimization Techniques",
    "simulation": "Simulation-Based Analysis",
    "routing": "Routing Algorithms",
    "network": "Network Science Approaches",
    "molecular": "Molecular Network Analysis",
    "protein": "Protein Interaction Modeling"
}


# -----------------------------
# Heading Recommendation
# -----------------------------
def recommend_journal_headings(user_semantics, top_k=3):
    if not user_semantics or not user_semantics.get("keywords"):
        return []

    domain = user_semantics.get("domain", "unknown")
    techniques = user_semantics.get("techniques", [])
    keywords = user_semantics.get("keywords", [])
    domain_title = DOMAIN_TITLES.get(domain, DOMAIN_TITLES["unknown"])

    phrases = []

    # Priority: techniques
    for t in techniques:
        for key, phrase in TECHNIQUE_MAP.items():
            if key in t:
                phrases.append(phrase)

    # Backup: keywords
    for k in keywords:
        for key, phrase in TECHNIQUE_MAP.items():
            if key in k:
                phrases.append(phrase)

    # Deduplicate while preserving order
    phrases = list(dict.fromkeys(phrases))

    # Fallback if nothing matched
    if not phrases:
        phrases = ["Computational Methods"]

    headings = []

    for p in phrases[:top_k]:
        headings.append(f"{p} for {domain_title}")

    return headings