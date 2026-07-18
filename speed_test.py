import time
import requests

API_URL = "http://127.0.0.1:8000/analyze"

payloads = [
    {
        "title": "Deep Learning Optimization",
        "abstract": "We propose a deep learning based network optimization system for IoT applications.",
    },
    {
        "title": "Graph Neural Networks",
        "abstract": "This paper explores graph neural networks for communication optimization in distributed systems.",
    },
    {
        "title": "Cloud Computing Systems",
        "abstract": "We analyze scalable cloud systems using distributed architectures and adaptive scheduling.",
    },
]

for i, payload in enumerate(payloads):
    start = time.time()

    try:
        res = requests.post(API_URL, json=payload, timeout=30)
    except Exception as e:
        print(f"\nTest {i + 1}")
        print("Request failed:", str(e))
        continue

    end = time.time()

    print(f"\nTest {i + 1}")
    print("Time:", round(end - start, 2), "seconds")

    try:
        data = res.json()

        # ---- HANDLE ERROR RESPONSE ----
        if data.get("status") == "ERROR":
            print("API ERROR:", data.get("message"))
            continue

        # ---- NORMAL CASE ----
        recommendation = data.get("final_recommendation", "MISSING")
        confidence = data.get("best_confidence", "N/A")
        journal = data.get("best_journal", "N/A")

        print("Recommendation:", recommendation)
        print("Journal:", journal)
        print("Confidence:", confidence)

        spread = data.get("spread", "N/A")
        print("Spread:", spread)

        # ---- DEBUG FALLBACK ----
        if recommendation in [None, "MISSING"]:
            print("[DEBUG FULL RESPONSE]:", data)

    except Exception as e:
        print("Error parsing response:", str(e))
        print("Raw response:", res.text)
