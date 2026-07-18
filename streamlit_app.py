import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Journal Recommendation Engine", layout="centered")
st.title("Journal Recommendation Engine")

# ---------------- PHASE 1: TITLE CHECK ----------------
st.header("Step 1: Enter Article Title")

if "title_ok" not in st.session_state:
    st.session_state.title_ok = False

title = st.text_input(
    "Proposed Article Title", placeholder="Enter your article title here..."
)

if st.button("Check Title"):
    if not title.strip():
        st.error("Title cannot be empty")
    else:
        with st.spinner("Checking title..."):
            resp = requests.post(f"{API_BASE}/check-title", json={"title": title})

        if resp.status_code != 200:
            st.error("Title check API error")
        else:
            res = resp.json()
            status = res.get("status")

            if status == "EXACT_MATCH":
                st.error(
                    f"❌ Title already exists in dataset "
                    f"(confidence: {round(res.get('confidence', 0) * 100, 1)}%)."
                )
                st.session_state.title_ok = False

            elif status == "NEAR_MATCH":
                st.warning(
                    f"⚠️ Similar title exists "
                    f"(confidence: {round(res.get('confidence', 0) * 100, 1)}%). "
                    "Consider revising."
                )
                st.session_state.title_ok = False

            else:
                st.success("✅ Title is acceptable. You may proceed.")
                st.session_state.title_ok = True

# ---------------- PHASE 2: ABSTRACT ANALYSIS ----------------
if st.session_state.title_ok:
    st.divider()
    st.header("Step 2: Paste Abstract")

    abstract = st.text_area(
        "Article Abstract",
        height=200,
        placeholder="Paste your research abstract here...",
    )

    if st.button("Analyze Abstract"):
        if not abstract.strip():
            st.error("Abstract cannot be empty")
            st.stop()

        with st.spinner("Analyzing abstract..."):
            response = requests.post(
                f"{API_BASE}/analyze", json={"title": title, "abstract": abstract}
            )

        if response.status_code != 200:
            st.error("Analysis API error")
            st.stop()

        result = response.json()

        # ---------- DUPLICATION HANDLING ----------
        if result.get("status") == "EXACT_MATCH":
            st.error(
                f"❌ Abstract already exists "
                f"(duplication confidence: {round(result.get('duplication_confidence', 0) * 100, 1)}%)."
            )
            st.stop()

        if result.get("status") == "NEAR_DUPLICATE":
            st.warning(
                f"⚠️ Abstract is highly similar to existing work "
                f"(duplication confidence: {round(result.get('duplication_confidence', 0) * 100, 1)}%)."
            )

        # ---------- RAG EXPLANATION + VALIDATOR OVERRIDE ----------
        rag = result.get("rag_analysis") or result.get("rag_explanations")

        st.subheader("AI Explanation")

        if rag and isinstance(rag, dict):
            global_exp = rag.get("global_explanation") or rag.get("global")
            pipeline_agreement = rag.get("pipeline_agreement", True)
            llm_pick = rag.get("llm_pick")
            override = rag.get("override", False)

            if isinstance(global_exp, dict):
                best_journal = global_exp.get("best_journal", "Unknown")
                reason = global_exp.get("reason", "")

                if reason:
                    st.success(f"Recommended Journal: {best_journal}")
                    st.write(f"Reason: {reason}")

                # -------- VALIDATOR OVERRIDE SIGNAL --------
                # This is the key contribution — when LLM disagrees with
                # the retrieval pipeline, surface it prominently.
                if not pipeline_agreement and llm_pick:
                    ranked = result.get("top3_recommendations", [])
                    pipeline_top = (
                        ranked[0].get("journal_name", "Unknown")
                        if ranked
                        else "Unknown"
                    )

                    if override:
                        st.warning(
                            f"⚠️ **AI Reviewer suggests a different venue**: "
                            f"**{llm_pick}** — this journal is not in the current "
                            f"dataset but may be a better fit based on domain analysis."
                        )
                    else:
                        st.info(
                            f"💡 **AI Reviewer disagrees with pipeline ranking**: "
                            f"Pipeline top pick was **{pipeline_top}**, but the AI Reviewer "
                            f"independently selected **{llm_pick}** as a better fit."
                        )
            else:
                st.warning("Invalid AI response format.")
        else:
            st.info("AI explanation not generated.")

        # ---------- OUTPUT ----------
        ranked = result.get("top3_recommendations", [])

        if ranked:
            ranked = sorted(
                ranked, key=lambda x: x.get("confidence", 0.0), reverse=True
            )

            st.subheader("Journal Confidence Scores")
            journals = [j.get("journal_name", "Unknown") for j in ranked]
            confidences = [j.get("confidence", 0.0) * 100 for j in ranked]
            st.bar_chart(dict(zip(journals, confidences)))

            # ---------- CONFIDENCE TABLE ----------
            st.subheader("Detailed Scores Table")
            df_display = pd.DataFrame(ranked).sort_values(
                by="confidence", ascending=False
            )

            if not df_display.empty:
                top = df_display.iloc[0]
                st.success(
                    f"{top['journal_name']} (confidence: {round(top['confidence'], 3)})"
                )

            display_cols = [
                c
                for c in ["journal_name", "confidence", "similarity"]
                if c in df_display.columns
            ]
            df_display["confidence"] = df_display["confidence"] * 100
            st.dataframe(
                df_display[display_cols].rename(
                    columns={"confidence": "confidence (%)"}
                )
            )

            # ---------- COLOR SIGNAL ----------
            st.subheader("Confidence Signal")
            top_conf = max(confidences) if confidences else 0.0

            if top_conf >= 50:
                st.success(f"🟢 Strong Match ({round(top_conf, 1)}%)")
            elif top_conf >= 35:
                st.warning(f"🟡 Moderate Match ({round(top_conf, 1)}%)")
            else:
                st.error(f"🔴 Weak Match ({round(top_conf, 1)}%)")

            # ---------- TOP JOURNAL RECOMMENDATIONS ----------
            st.subheader("Top Journal Recommendations")

            for j in ranked:
                journal_name = j.get("journal_name", "Unknown")
                sim = round(j.get("similarity", 0.0), 3)
                conf = round(j.get("confidence", 0.0) * 100, 1)

                st.markdown(f"### {journal_name}")
                st.write(f"Confidence: {conf}%")
                st.write(f"Similarity: {sim}")

                # Use dynamic reason from final_decision if available,
                # else fall back to similarity-based text
                reason = j.get("reason", "")
                if not reason:
                    explanation = j.get("explanation", {})
                    if isinstance(explanation, dict):
                        reason = explanation.get("reason", "")

                if not reason:
                    if sim >= 0.65:
                        reason = "Strong alignment between the paper's methodology and the journal's scope."
                    elif sim >= 0.45:
                        reason = "Moderate relevance based on topic similarity and research direction."
                    else:
                        reason = "Limited alignment with the journal's focus area."

                st.write(f"Reason: {reason}")
                st.markdown("---")

            # ---------- DOWNLOAD REPORT ----------
            st.subheader("Export Report")

            rag_summary = ""
            if rag and isinstance(rag, dict):
                global_exp = rag.get("global_explanation") or rag.get("global")
                if isinstance(global_exp, dict):
                    rag_summary = (
                        f"\nAI Reviewer Recommendation: {global_exp.get('best_journal', 'N/A')}"
                        f"\nReason: {global_exp.get('reason', 'N/A')}"
                        f"\nPipeline Agreement: {rag.get('pipeline_agreement', True)}"
                    )

            report_text = (
                f"Title: {title}\n\n"
                f"Abstract:\n{abstract}\n"
                f"{rag_summary}\n\n"
                f"Top Journals:\n"
            )
            for j in ranked:
                report_text += (
                    f"- {j.get('journal_name')} | "
                    f"Confidence: {round(j.get('confidence', 0.0) * 100, 1)}% | "
                    f"Similarity: {round(j.get('similarity', 0.0), 3)}\n"
                )

            st.download_button(
                label="Download Report",
                data=report_text,
                file_name="journal_recommendation.txt",
                mime="text/plain",
            )

        with st.expander("Detailed Results"):
            st.json(result)
