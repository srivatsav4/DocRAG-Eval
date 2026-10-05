from __future__ import annotations

SYSTEM_PROMPT = (
    "You answer questions about PostgreSQL using ONLY the numbered evidence passages provided. "
    "Rules: (1) Every factual sentence must end with a citation like [S1] or [S1, S2]. "
    "(2) Do not use outside knowledge. (3) If the evidence only partially answers the question, answer the supported "
    "part and state what is missing. (4) Be concise."
)


def build_user_prompt(question: str, evidence: list[tuple[str, str, str]], limited: bool = False) -> str:
    """evidence: list of (label, breadcrumb, text)."""
    blocks = "\n\n".join(f"[{label}] {crumb}\n{text}" for label, crumb, text in evidence)
    note = ("\nNOTE: the evidence is only partially relevant; explicitly state any limitations.\n" if limited else "")
    return f"Evidence:\n{blocks}\n{note}\nQuestion: {question}\n\nAnswer (with [S#] citations):"
