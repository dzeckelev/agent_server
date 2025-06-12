# agent_server.py
"""
Minimal FastAPI + CrewAI gateway that calls your local Ollama model and

Environment`
-----------
SERPER_API_KEY   – required, get one free at https://serper.dev
OLLAMA_HOST      – default http://127.0.0.1:11434
OLLAMA_MODEL     – default codestral:22b-q6_K_M
"""

import os
from time import time
from typing import Any, Coroutine

import anyio
from fastapi import FastAPI
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, LLM, CrewOutput, Process
from crewai_tools import SerperDevTool
from ollama import Client
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ------------------------------------------------------------------ CONFIG ---
SERPER_API_KEY = os.environ["SERPER_API_KEY"]  # raises KeyError if   missing
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/codestral:22b")

ollama_llm = LLM(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    # llm_provider="ollama",
)
search_tool = SerperDevTool()

# ---------------------------------------------------------- FASTAPI APP -------
app = FastAPI(title="CrewAI-Gateway", version="0.1")

# allow IDE WebView origins (localhost / 127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel, extra="allow"):
    prompt: str | None = None
    messages: list[dict] | None = None
    stream: bool | None = False
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/healthz", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", tags=["chat"])
async def completions(req: ChatRequest):
    # unify input
    if req.messages:
        prompt_text = req.messages[-1]["content"]
    elif req.prompt:
        prompt_text = req.prompt
    else:
        raise ValueError("No prompt or messages provided")

    def run(prompt: str) -> str:
        # create fresh agents each call – they are lightweight
        planner = Agent(
            role='Planner',
            goal='Read the user request. Decide whether we need a full runnable solution or a partial refactor. Break the job into research + coding steps, list acceptance criteria (compiles, tests pass, linter clean), and assign work to the Researcher and Developer. Do not write code yourself.',
            backstory='A veteran tech lead who has managed dozens of successful projects. Great at turning fuzzy ideas into step-by-step plans and communicating them clearly to the team.',
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False,
        )

        # Create your researcher agent with Ollama
        researcher = Agent(
            role='Researcher',
            goal='Use web search to collect 3–5 up-to-date code snippets, docs, or blog posts that directly address the Planner’s plan. Summarise findings in short bullets with URLs. Do not write final code.',
            backstory='An investigative developer who loves digging through docs, RFCs, GitHub issues and Stack Overflow to find the freshest, most reliable nuggets of information.',
            tools=[search_tool],
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False,
        )

        coder = Agent(
            role='Developer',
            goal='Using the Planner’s criteria and Researcher’s notes, produce the requested code. If a full example is needed, output a complete runnable file; if a refactor, change only the selected block and leave the rest intact. No Markdown',
            backstory='A pragmatic senior engineer who writes clean, idiomatic code, keeps imports tidy, and refuses to over-engineer.',
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False,
        )

        planner_task = Task(
            description=f"Create a clear step-by-step plan for: {prompt}",
            expected_output='A bullet-point plan listing research queries, coding steps, acceptance criteria, and which agent handles each step. No code.',
            agent=planner
        )

        research_task = Task(
            description='Fetch the resources the Planner requested and summarise them.',
            expected_output='Bullet list of key findings, each with a working URL or repo link, plus one-sentence relevance note. No code.',
            agent=researcher,
            context=[planner_task]
        )

        implementation_task = Task(
            description='Implement the solution or refactor according to the plan and research notes.',
            expected_output=(
                "A) For full solutions: a complete runnable source file "
                "with all imports and minimal inline comments, formatted by the "
                "language’s standard tool.  "
                "B) For refactors: only the edited code block, unchanged context lines.  "
                "No Markdown fences."
            ),
            agent=coder,
            context=[planner_task, research_task],
        )

        crew = Crew(
            agents=[planner, researcher, coder],
            tasks=[planner_task, research_task, implementation_task],
            process=Process.sequential,
            verbose=True,
            # planning=True,
            # planning_llm=ollama_llm,
        )
        return str(crew.kickoff())  # ensure str

    answer = await anyio.to_thread.run_sync(run, prompt_text)
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

    # even if Continue sent "stream":true -> give plain JSON
    return JSONResponse(
        content=payload,
        headers={
            "X-Accel-Buffering": "no",       # avoid proxy buffering
        },
    )
