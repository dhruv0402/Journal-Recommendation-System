import re

file_path = "/Users/dhruvgourisaria/JournalProject/build_paper_v2.js"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Table 2 metrics
content = content.replace(
    '["BM25 (baseline\\u2020)", "56.4%", "90.1%", "0.734", "0.766"]',
    '["BM25 (baseline\\u2020)", "93.8%", "94.1%", "0.939", "0.940"]'
)
content = content.replace(
    '["FAISS-only", "11.7%", "65.9%", "0.413", "0.423"]',
    '["FAISS-only", "20.5%", "71.1%", "0.472", "0.494"]'
)
content = content.replace(
    '["Full pipeline", "48.7%", "87.5%", "0.685", "0.718"]',
    '["Full pipeline", "76.2%", "93.0%", "0.843", "0.863"]'
)
content = content.replace(
    '["Full pipeline + LLM validator", "56.4%", "87.5%", "0.732", "0.752"]',
    '["Full pipeline + LLM validator", "63.0%", "93.0%", "0.780", "0.818"]'
)

# 2. Update Table 3 overrides
content = content.replace(
    '["Total overrides triggered", "151", "55.3% of test set"]',
    '["Total overrides triggered", "108", "39.6% of test set"]'
)
content = content.replace(
    '["Overrides that improved Hit@1", "73", "48.3% precision"]',
    '["Overrides that improved Hit@1", "29", "26.9% precision"]'
)
content = content.replace(
    '["Overrides that hurt Hit@1", "52", "34.4% of overrides"]',
    '["Overrides that hurt Hit@1", "65", "60.2% of overrides"]'
)
content = content.replace(
    '["Overrides with no change (both wrong)", "26", "17.2% of overrides"]',
    '["Overrides with no change (both wrong)", "14", "13.0% of overrides"]'
)

# 3. Update Table 4 per-journal
content = content.replace(
    '["Eng. Applications of AI (EAAI)",          "50", "94.0%",  "100.0%", "0.967"]',
    '["Eng. Applications of AI (EAAI)",          "50", "100.0%", "100.0%", "1.000"]'
)
content = content.replace(
    '["Journal of Systems and Software",         "45", "82.2%",  "97.8%",  "0.889"]',
    '["Journal of Systems and Software",         "45", "95.6%",  "100.0%", "0.978"]'
)
content = content.replace(
    '["Computer Networks",                       "50", "64.0%",  "100.0%", "0.817"]',
    '["Computer Networks",                       "50", "88.0%",  "100.0%", "0.937"]'
)
content = content.replace(
    '["Advances in Engineering Software",        "24", "33.3%",  "100.0%", "0.653"]',
    '["Advances in Engineering Software",        "24", "79.2%",  "100.0%", "0.889"]'
)
content = content.replace(
    '["Computer Vision and Image Understanding", "44", "11.4%",  "97.7%",  "0.536"]',
    '["Computer Vision and Image Understanding", "44", "75.0%",  "97.7%",  "0.869"]'
)
content = content.replace(
    '["Parallel Computing",                      "11", "27.3%",  "72.7%",  "0.513"]',
    '["Parallel Computing",                      "11", "63.6%",  "100.0%", "0.788"]'
)
content = content.replace(
    '["Artificial Intelligence",                 "15", "6.7%",   "46.7%",  "0.283"]',
    '["Artificial Intelligence",                 "15", "46.7%",  "100.0%", "0.711"]'
)
content = content.replace(
    '["Computer Standards & Interfaces",         "8",  "0.0%",   "25.0%",  "0.217"]',
    '["Computer Standards & Interfaces",         "8",  "37.5%",  "100.0%", "0.625"]'
)
content = content.replace(
    '["Journal of Computer and System Sciences", "3",  "0.0%",   "33.3%",  "0.244"]',
    '["Journal of Computer and System Sciences", "3",  "0.0%",   "33.3%",  "0.317"]'
)
content = content.replace(
    '["AI Open",                                 "7",  "0.0%",   "14.3%",  "0.176"]',
    '["AI Open",                                 "7",  "28.6%",  "100.0%", "0.619"]'
)
content = content.replace(
    '["Computer Communications",                 "16", "0.0%",   "56.2%",  "0.320"]',
    '["Computer Communications",                 "16", "0.0%",   "0.0%",   "0.000"]'
)

# 4. Update the text paragraphs
old_p1 = 'The full pipeline with LLM validator achieves strong performance, with Hit@3 of 87.5% and MRR of 0.732. The LLM validator contributes a 382.1% relative improvement in Hit@1 over the FAISS-only baseline (11.7% to 56.4%), demonstrating that LLM-based validation substantially corrects for retrieval errors that the reranking components cannot address. The dual reranking components contribute a more modest but consistent improvement in Hit@3 (65.9% to 87.5%), primarily by resolving near-ties between journals with similar aggregated similarity scores.'
new_p1 = 'The full pipeline achieves strong performance, with Hit@3 of 93.0% and MRR of 0.843. The dual reranking components contribute a significant relative improvement in Hit@1 over the FAISS-only baseline (20.5% to 76.2%), demonstrating that the custom rerankers successfully resolve retrieval errors. The LLM validator over-corrects, dropping Hit@1 to 63.0%, indicating that a zero-shot LLM validator struggles to outperform the custom-trained learning reranker on this domain-specific dataset.'
content = content.replace(old_p1, new_p1)

old_p1_alt = 'Notably, neither the full pipeline nor the full pipeline with validator improves Hit@3 beyond 87.5%, suggesting that for the remaining 12.5% of test cases, the correct journal is consistently ranked below position 3 across all conditions. Per-journal analysis (Section 5.4) reveals that these errors are concentrated in journals with high topical overlap and large corpus size, where semantic similarity alone is insufficient to resolve the recommendation.'
new_p1_alt = 'Notably, neither the full pipeline nor the full pipeline with validator improves Hit@3 beyond 93.0%, suggesting that for the remaining 7.0% of test cases, the correct journal is consistently ranked below position 3 across all conditions. Per-journal analysis (Section 5.4) reveals that these errors are concentrated in journals with high topical overlap and large corpus size, where semantic similarity alone is insufficient to resolve the recommendation.'
content = content.replace(old_p1_alt, new_p1_alt)

old_p2 = 'The validator triggered overrides in 55.3% of test cases. Of these 151 overrides, 73 resulted in an improvement to Hit@1 (the validator\'s preferred journal was correct and was promoted to rank 1), 52 resulted in a degradation (the validator\'s preferred journal was incorrect and the pipeline\'s original rank-1 journal was correct), and 26 produced no change to Hit@1 (neither the pipeline\'s top pick nor the validator\'s preferred journal was the correct journal). The resulting override precision of 48.3% — defined as the proportion of overrides that improved Hit@1 — confirms that when the LLM\'s domain assessment disagrees with the retrieval ranking, it is correct nearly half the time.'
new_p2 = 'The validator triggered overrides in 39.6% of test cases. Of these 108 overrides, 29 resulted in an improvement to Hit@1, 65 resulted in a degradation, and 14 produced no change. The resulting override precision of 26.9% indicates that the validator overrules the pipeline in 108 cases, but in most cases (65 out of 108), it hurts the Hit@1 score, resulting in a net decrease of 36 correct predictions.'
content = content.replace(old_p2, new_p2)

old_p3 = 'The majority journals — Engineering Applications of Artificial Intelligence (94.0% Hit@1), Journal of Systems and Software (82.2% Hit@1), and Computer Networks (64.0% Hit@1) — show exceptional performance.'
new_p3 = 'The majority journals — Engineering Applications of Artificial Intelligence (100.0% Hit@1), Journal of Systems and Software (95.6% Hit@1), and Computer Networks (88.0% Hit@1) — show exceptional performance.'
content = content.replace(old_p3, new_p3)

old_p4 = 'Conversely, the minority journals — including AI Open, JCSS, and Computer Standards & Interfaces — achieve 0.0% Hit@1 under the full pipeline.'
new_p4 = 'Conversely, the minority journals — including AI Open, JCSS, and Computer Standards & Interfaces — achieve lower Hit@1 under the full pipeline.'
content = content.replace(old_p4, new_p4)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete inside build_paper_v2.js")
