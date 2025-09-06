"""
LLM Interview Simulator — Streamlit (Starter App)
-------------------------------------------------
A lightweight, extensible app that simulates technical or behavioral interviews,
asks role/domain-specific questions, evaluates responses with rubric-driven scoring,
and produces a final summary with strengths, improvements, and resources.

Quick start
-----------
1) pip install -U streamlit openai python-dotenv pandas tiktoken
2) Create a .env file with: OPENAI_API_KEY=sk-...
3) Run: streamlit run app.py

Notes
-----
- Default model/provider: OpenAI (gpt-4o-mini); swap via the `LLM` class if desired.
- Data is stored in-memory (st.session_state) and exportable as JSON/CSV from the UI.
- PDF export stub included for future extension.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ------------------------------
# LLM Provider Abstraction
# ------------------------------
try:
    from openai import OpenAI  # OpenAI SDK v1.x
except Exception:
    OpenAI = None


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 600


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = None
        if cfg.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai`. ")
            self.client = OpenAI()
        else:
            raise NotImplementedError("Only OpenAI provider is implemented in this starter.")

    def complete(self, system: str, user: str) -> str:
        if self.cfg.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        else:
            raise NotImplementedError


# ------------------------------
# Prompt Templates
# ------------------------------
QUESTION_SYS = (
    "You are an expert interviewer. Generate concise, role-appropriate interview questions. "
    "Return ONLY a JSON list of strings with no extra prose."
)

QUESTION_USER_TMPL = (
    "Role: {role}\nDomain: {domain}\nMode: {mode}\nCount: {count}\n"
    "Constraints: questions should be progressively challenging, clear, and test real-world skills. "
    "For technical: include a mix from algorithms/coding, system design, or domain-specific knowledge as appropriate. "
    "For behavioral: prefer STAR prompts and scenarios."
)

EVAL_SYS = (
    "You are a meticulous interviewer and grader. Evaluate the candidate's answer against the rubric. "
    "Be concise, fair, and specific. Respond ONLY as a JSON object with fields: "
    "{score: number (0-10), feedback: string, criteria: {clarity:0-10, correctness:0-10, completeness:0-10, "
    "examples:0-10 (behavioral only or 0 if NA), technical_accuracy:0-10 (technical only or 0 if NA)}}."
)

EVAL_USER_TMPL = (
    "Mode: {mode}\nRole: {role}\nDomain: {domain}\nQuestion: {question}\nAnswer: {answer}\n"
    "Rubric: Clarity, Correctness, Completeness; include Technical Accuracy for technical OR Examples/STAR for behavioral.\n"
    "Return JSON only."
)

SUMMARY_SYS = (
    "You are an expert career coach. Summarize the interview session. Return ONLY a JSON object with fields: "
    "{strengths: [..], improvements: [..], overall_feedback: string, resources: [ {title, url?, note} ], score_out_of_10: number }."
)

SUMMARY_USER_TMPL = (
    "Mode: {mode}\nRole: {role}\nDomain: {domain}\nItems: {items}\n"
    "Each item has question, user_answer, score(0-10), criteria. Weigh all answers and produce actionable insights."
)

# ------------------------------
# Data Structures
# ------------------------------
@dataclass
class QARecord:
    index: int
    question: str
    user_answer: str
    evaluation: Dict[str, Any]


# ------------------------------
# Question Generation
# ------------------------------

def generate_questions(llm: LLM, role: str, domain: str, mode: str, count: int) -> List[str]:
    user = QUESTION_USER_TMPL.format(role=role, domain=domain, mode=mode, count=count)
    raw = llm.complete(QUESTION_SYS, user)
    # Best-effort parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data][:count]
    except Exception:
        pass
    # Fallback: split lines
    return [s.strip("- •\n ") for s in raw.split("\n") if s.strip()][:count]


# ------------------------------
# Evaluation
# ------------------------------

def evaluate_answer(llm: LLM, mode: str, role: str, domain: str, question: str, answer: str) -> Dict[str, Any]:
    user = EVAL_USER_TMPL.format(mode=mode, role=role, domain=domain, question=question, answer=answer)
    raw = llm.complete(EVAL_SYS, user)
    try:
        data = json.loads(raw)
        # Minimal validation
        _ = data.get("score", 0)
        _ = data.get("feedback", "")
        _ = data.get("criteria", {})
        return data
    except Exception:
        # Fallback minimal structure
        return {
            "score": 0,
            "feedback": f"(Parsing issue) Model returned: {raw[:500]}",
            "criteria": {"clarity": 0, "correctness": 0, "completeness": 0, "examples": 0, "technical_accuracy": 0},
        }


# ------------------------------
# Summary
# ------------------------------

def summarize_session(llm: LLM, mode: str, role: str, domain: str, records: List[QARecord]) -> Dict[str, Any]:
    items = [
        {
            "question": r.question,
            "user_answer": r.user_answer,
            "score": r.evaluation.get("score", 0),
            "criteria": r.evaluation.get("criteria", {}),
        }
        for r in records
    ]
    user = SUMMARY_USER_TMPL.format(mode=mode, role=role, domain=domain, items=json.dumps(items))
    raw = llm.complete(SUMMARY_SYS, user)
    try:
        data = json.loads(raw)
        return data
    except Exception:
        return {
            "strengths": [],
            "improvements": [],
            "overall_feedback": f"(Parsing issue) Model returned: {raw[:500]}",
            "resources": [],
            "score_out_of_10": round(sum(x["score"] for x in items)/max(1, len(items)), 2),
        }


# ------------------------------
# Helpers
# ------------------------------

def init_state():
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("current_idx", 0)
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("mode", "Technical Interview")
    st.session_state.setdefault("role", "Software Engineer")
    st.session_state.setdefault("domain", "General")
    st.session_state.setdefault("count", 4)


def reset_session():
    st.session_state.started = False
    st.session_state.questions = []
    st.session_state.current_idx = 0
    st.session_state.records = []


# ------------------------------
# UI
# ------------------------------

def main():
    load_dotenv()
    st.set_page_config(page_title="LLM Interview Simulator", page_icon="🎤", layout="wide")

    st.title("🎤 LLM Interview Simulator")
    st.caption("Practice technical or behavioral interviews with instant, rubric-based feedback.")

    with st.sidebar:
        st.header("Setup")
        provider = st.selectbox("Provider", ["openai"], index=0)
        model = st.text_input("Model", value="gpt-4o-mini", help="Swap to gpt-4o or gpt-4.1 if available.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max tokens (eval/summary)", 256, 2000, 600, 32)

        st.divider()
        st.subheader("Interview Config")
        mode = st.selectbox("Mode", ["Technical Interview", "Behavioral Interview"], index=0)
        role = st.selectbox("Target Role", [
            "Software Engineer", "Frontend Engineer", "Backend Engineer", "Data Analyst", "Data Scientist",
            "ML Engineer", "Product Manager", "DevOps Engineer", "SRE", "QA Engineer",
        ], index=0)
        domain = st.text_input("Domain / Focus (optional)", value="General", placeholder="e.g., React, System Design, SQL, ML, Leadership")
        count = st.slider("Questions", 3, 7, 4)

        st.divider()
        if st.button("🔄 Reset Session"):
            reset_session()

    init_state()

    # Build LLM
    cfg = LLMConfig(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
    try:
        llm = LLM(cfg)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Start / Generate questions
    if not st.session_state.started:
        st.info("Configure options in the sidebar, then click **Start Interview**.")
        if st.button("🚀 Start Interview", type="primary"):
            qs = generate_questions(llm, role, domain, mode, count)
            st.session_state.questions = qs
            st.session_state.mode = mode
            st.session_state.role = role
            st.session_state.domain = domain
            st.session_state.count = count
            st.session_state.started = True
            st.rerun()
        return

    # Interview in progress
    q_idx = st.session_state.current_idx
    questions = st.session_state.questions
    mode = st.session_state.mode
    role = st.session_state.role
    domain = st.session_state.domain

    # Header progress
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"Question {q_idx + 1} of {len(questions)}")
    with col2:
        st.metric("Answered", f"{len(st.session_state.records)}")
    with col3:
        avg = (
            sum([r.evaluation.get("score", 0) for r in st.session_state.records]) / max(1, len(st.session_state.records))
            if st.session_state.records else 0
        )
        st.metric("Avg Score", f"{avg:.1f} / 10")

    if q_idx < len(questions):
        question = questions[q_idx]
        st.write(f"**{question}**")
        answer = st.text_area("Your answer", key=f"answer_{q_idx}", height=180, placeholder="Type your response here...")

        c1, c2, c3, c4 = st.columns([1,1,1,2])
        with c1:
            submit = st.button("Submit", type="primary")
        with c2:
            retry = st.button("Retry")
        with c3:
            skip = st.button("Skip")
        with c4:
            end_now = st.button("End Interview")

        if submit:
            if not answer.strip():
                st.warning("Please enter an answer before submitting.")
            else:
                eval_json = evaluate_answer(llm, mode, role, domain, question, answer)
                rec = QARecord(index=q_idx, question=question, user_answer=answer, evaluation=eval_json)
                st.session_state.records.append(rec)
                st.session_state.current_idx += 1
                st.rerun()

        if retry:
            st.experimental_rerun()

        if skip:
            # Record empty answer with minimal eval (score 0)
            eval_json = {"score": 0, "feedback": "Skipped.", "criteria": {"clarity": 0, "correctness": 0, "completeness": 0, "examples": 0, "technical_accuracy": 0}}
            rec = QARecord(index=q_idx, question=question, user_answer="(skipped)", evaluation=eval_json)
            st.session_state.records.append(rec)
            st.session_state.current_idx += 1
            st.rerun()

        if end_now:
            st.session_state.current_idx = len(questions)
            st.rerun()

    # Completed
    if st.session_state.current_idx >= len(questions):
        st.success("Interview completed! Review your results below.")
        records: List[QARecord] = st.session_state.records

        # Table of Q/A
        table_rows = []
        for r in records:
            crit = r.evaluation.get("criteria", {})
            table_rows.append({
                "#": r.index + 1,
                "Question": r.question,
                "Your Answer": r.user_answer,
                "Score": r.evaluation.get("score", 0),
                "Clarity": crit.get("clarity", 0),
                "Correctness": crit.get("correctness", 0),
                "Completeness": crit.get("completeness", 0),
                "Examples": crit.get("examples", 0),
                "Tech Acc": crit.get("technical_accuracy", 0),
                "Feedback": r.evaluation.get("feedback", ""),
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Overall summary
        if st.button("🧾 Generate Final Summary", type="primary"):
            with st.spinner("Summarizing..."):
                summary = summarize_session(llm, st.session_state.mode, st.session_state.role, st.session_state.domain, records)
                st.session_state["summary"] = summary
                st.rerun()

        summary = st.session_state.get("summary")
        if summary:
            st.subheader("Final Summary")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Strengths**")
                for s in summary.get("strengths", []) or []:
                    st.write(f"- {s}")
            with colB:
                st.markdown("**Areas to Improve**")
                for s in summary.get("improvements", []) or []:
                    st.write(f"- {s}")
            st.markdown(f"**Overall Feedback:** {summary.get('overall_feedback', '')}")
            st.markdown(f"**Final Score:** {summary.get('score_out_of_10', 0)} / 10")

            res = summary.get("resources", []) or []
            if res:
                st.markdown("**Suggested Resources**")
                for r in res:
                    title = r.get("title") or "Resource"
                    url = r.get("url")
                    note = r.get("note")
                    if url:
                        st.markdown(f"- [{title}]({url}) — {note or ''}")
                    else:
                        st.markdown(f"- {title} — {note or ''}")

            # Exports
            st.divider()
            st.subheader("Export")
            export = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": st.session_state.mode,
                "role": st.session_state.role,
                "domain": st.session_state.domain,
                "questions": st.session_state.questions,
                "records": [asdict(r) for r in records],
                "summary": summary,
            }
            json_bytes = json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Download JSON", data=json_bytes, file_name="interview_session.json", mime="application/json")

            # CSV export (flat)
            csv_df = pd.DataFrame([
                {
                    "index": r.index + 1,
                    "question": r.question,
                    "user_answer": r.user_answer,
                    "score": r.evaluation.get("score", 0),
                    **{f"criteria_{k}": v for k, v in (r.evaluation.get("criteria", {}) or {}).items()},
                    "feedback": r.evaluation.get("feedback", ""),
                }
                for r in records
            ])
            st.download_button("⬇️ Download CSV", data=csv_df.to_csv(index=False).encode("utf-8"), file_name="interview_results.csv", mime="text/csv")

            # PDF stub (left as a TODO for adding reportlab or html->pdf)
            st.caption("PDF export: add reportlab or weasyprint if you want direct PDF generation.")

    # Footer
    st.divider()
    with st.expander("⚙️ Scoring rubric details"):
        st.markdown(
            """
            **Technical interviews** weigh Clarity, Correctness, Completeness, and Technical Accuracy.
            **Behavioral interviews** weigh Clarity, Correctness, Completeness, and Use of Examples/STAR.
            Each answer is scored 0–10; weights are determined by the LLM using the rubric and question type.
            """
        )
    st.caption("Tip: Use the sidebar to adjust model/temperature and the number of questions.")


if __name__ == "__main__":
    main()
