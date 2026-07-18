"""
Static scope descriptions per journal.
Used by scope_reranker.py to boost confidence via scope-level semantic similarity.

Previously only had 6 entries — expanded to cover all major journals in the dataset.
The build_domain_map.py script can regenerate/extend this automatically.
"""

journal_scope_map = {
    # --- Networks & Communication ---
    "Computer Networks": "network routing communication protocols wireless systems distributed communication",
    "Journal of Network and Computer Applications": "network applications distributed systems internet protocols cloud computing",
    "Computer Communications": "telecommunications wireless networks protocols data transmission",
    "Ad Hoc Networks": "wireless ad hoc sensor networks mobile routing",
    "Wireless Networks": "wireless communication mobile networks signal processing",
    "IEEE Transactions on Networking": "network architecture protocols performance internet systems",

    # --- Systems & Architecture ---
    "Parallel Computing": "parallel systems distributed computing scheduling HPC performance optimization",
    "Journal of Systems and Software": "software engineering system design testing architecture software development",
    "Advances in Engineering Software": "software engineering tools modeling simulation engineering systems",
    "Journal of Parallel and Distributed Computing": "distributed systems parallel processing concurrency performance",
    "Future Generation Computer Systems": "cloud computing distributed systems scalability virtualization",
    "Computers and Electrical Engineering": "electrical engineering embedded systems signal processing hardware",

    # --- AI & Machine Learning ---
    "Artificial Intelligence": "machine learning AI reasoning knowledge systems neural networks",
    "AI Open": "AI data driven learning models machine intelligence",
    "Neural Networks": "deep learning neural architectures backpropagation training optimization",
    "Expert Systems with Applications": "expert systems applied AI decision support knowledge engineering",
    "Pattern Recognition": "pattern recognition image processing classification feature extraction",
    "Knowledge-Based Systems": "knowledge representation reasoning ontologies intelligent systems",
    "Neurocomputing": "neural computation learning algorithms recurrent networks",

    # --- Data & Databases ---
    "Information Sciences": "data science information systems knowledge management analytics",
    "Data and Knowledge Engineering": "databases knowledge engineering data modeling query processing",
    "Journal of Big Data": "big data analytics distributed storage data processing scalability",
    "Information Systems": "information systems database design enterprise systems",

    # --- Software Engineering ---
    "Software: Practice and Experience": "software development programming languages tools engineering practice",
    "Empirical Software Engineering": "software engineering empirical studies testing code quality",
    "Journal of Software: Evolution and Process": "software maintenance evolution refactoring process",

    # --- Security ---
    "Computers and Security": "cybersecurity intrusion detection cryptography network security",
    "Journal of Information Security and Applications": "information security privacy authentication access control",

    # --- Interdisciplinary / Applied ---
    "Applied Soft Computing": "soft computing evolutionary algorithms fuzzy logic optimization",
    "Swarm and Evolutionary Computation": "evolutionary computation swarm intelligence genetic algorithms",
    "Engineering Applications of Artificial Intelligence": "applied AI engineering optimization control systems",
    "Simulation Modelling Practice and Theory": "simulation modeling performance evaluation systems",
    "Journal of Computational Science": "computational science scientific computing numerical methods",
}
