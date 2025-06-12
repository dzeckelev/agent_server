"""
FastAPI gateway for CrewAI over a local Ollama LLM.

Environment variables
---------------------
SERPER_API_KEY   – required, get one free at https://serper.dev
OLLAMA_HOST      – default http://127.0.0.1:11434
OLLAMA_MODEL     – default codestral:22b-q6_K_M
"""

from __future__ import annotations

import os
import re
from time import time
from typing import List, Dict, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, LLM, Process
from crewai_tools import SerperDevTool
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ---------------------------------------------------------------- CONFIG ----
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    raise EnvironmentError("SERPER_API_KEY environment variable is required")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/codestral:22b")

ollama_llm = LLM(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024,
)

search_tool = SerperDevTool()

# --------------------------------------------------------- FASTAPI APP -----
app = FastAPI(title="CrewAI-Gateway", version="0.3.0")

# Allow IDE WebView origins explicitly; avoid wildcard in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------- Pydantic I/O ----
class ChatRequest(BaseModel):
    """Subset of OpenAI Chat Completions request"""

    messages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="OpenAI-style message list. The last element must be role=user.",
    )
    prompt: Optional[str] = Field(
        None, description="Fallback prompt if no messages provided.",
    )
    stream: bool = False
    temperature: float = 0.2
    max_tokens: Optional[int] = None

    @property
    def latest_user_message(self) -> str:
        if self.messages:
            return self.messages[-1]["content"]
        if self.prompt:
            return self.prompt
        raise HTTPException(422, detail="No prompt or messages provided.")


# ---------------------------------------------------- Message preprocessing ----
CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n([\s\S]+?)```", re.MULTILINE)


def split_instruction_and_code(msg: str) -> Tuple[str, Optional[str]]:
    """Return (instruction_text, code_block or None).

    If multiple code blocks are present, they are concatenated.
    """
    code_blocks = CODE_BLOCK_RE.findall(msg)
    code = "\n\n".join(code_blocks).strip() if code_blocks else None

    # Remove code blocks from instruction text
    instruction = CODE_BLOCK_RE.sub("", msg).strip()
    return instruction, code or None


# --------------------------------------------------------- Crew helpers ----

def _build_agents() -> list[Agent]:
    """Instantiate fresh, stateless agents per request."""

    planner = Agent(
        role="Planner",
        goal=(
            "Analyse the user request. Decide whether it requires a full runnable "
            "example or a refactor of the provided code snippet. Produce a numbered "
            "execution plan, research search terms, coding steps and acceptance "
            "criteria (tests, lint, build). Assign each step to the Researcher or "
            "Developer. Do NOT write any code."
        ),
        backstory="Staff engineer & tech‑lead with a knack for crystal‑clear roadmaps.",
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
    )

    researcher = Agent(
        role="Researcher",
        goal=(
            "Execute the web‑search queries from the plan. Collect up‑to‑date docs, "
            "blog posts, RFCs and GitHub issues. Return concise bullet points with "
            "the URL and one‑sentence relevance note. Do NOT write final code."
        ),
        backstory="Curious engineer who never stops reading release notes.",
        tools=[search_tool],
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
    )

    developer = Agent(
        role="Developer",
        goal=(
            "Produce the requested code.\n"
            "• If *code_snippet* was provided and the plan says 'refactor', emit a "
            "  unified diff for that snippet only.\n"
            "• Otherwise emit a complete runnable file.\n"
            "No Markdown, no extra commentary."
        ),
        backstory="Product‑minded senior developer who values readability & minimalism.",
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
    )

    return [planner, researcher, developer]


def _build_tasks(prompt: str, code: Optional[str], agents: list[Agent]) -> list[Task]:
    planner, researcher, developer = agents

    plan_task = Task(
        description=(
            "Draft a step‑by‑step execution plan for the following user request. "
            "If *code_snippet* is not None, treat it as a refactor request unless "
            "the instruction explicitly asks for a full rewrite.\n\n"
            f"Instruction:\n{prompt}\n\n"
            f"code_snippet:\n{code or '⟨none⟩'}"
        ),
        expected_output=(
            "• Numbered plan\n"
            "• Research search terms\n"
            "• Coding steps\n"
            "• Acceptance criteria\n"
            "• Explicit agent assignment per step"
        ),
        agent=planner,
    )

    research_task = Task(
        description="Gather the resources requested in the plan.",
        expected_output="• Bulleted list: [title] – URL – why relevant.",
        agent=researcher,
        context=[plan_task],
    )

    impl_task = Task(
        description=(
            "Implement the solution or refactor according to the plan and research "
            "notes. Use *code_snippet* as the basis if provided."
        ),
        expected_output=(
            "If full solution → full source file.\n"
            "If refactor → unified diff patch."
        ),
        agent=developer,
        context=[plan_task, research_task],
    )

    return [plan_task, research_task, impl_task]


def _run_crew(instruction_text: str, code_snippet: Optional[str]) -> str:
    agents = _build_agents()
    tasks = _build_tasks(instruction_text, code_snippet, agents)
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


# ----------------------------------------------------------- End‑point -----
@app.post("/v1/chat/completions", tags=["chat"])
async def completions(req: ChatRequest):
    instruction, code_snippet = split_instruction_and_code(req.latest_user_message)

    answer = await anyio.to_thread.run_sync(_run_crew, instruction, code_snippet)

    payload = {
        "id": f"agent-http-{int(time()*1000)}",
        "object": "chat.completion",
        "created": int(time()),
        "model": "agent-http",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # Even if Continue sent stream=true – respond with plain JSON for now
    return JSONResponse(
        content=payload,
        headers={"X-Accel-Buffering": "no"},
    )
