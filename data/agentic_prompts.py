"""Verbatim prompts from upstream agentic benchmarks.

Each constant is copied character-for-character from the upstream source so the
text the model sees matches what the benchmark authors intended. Sources are
cited inline above each constant.

# Skip-tools compromise (read before extending any agentic loader)

Across all three benchmarks (tau-bench, AgentBench-OS/KG/DB, AgentInstruct)
we make the same deliberate compromise:

  - We KEEP the official agent system prompts verbatim, including their
    abstract references to tool calls — e.g. tau-bench's "Make a tool call",
    AgentBench-OS's "call exactly one of the provided tools" /
    "finish_action" / "answer_action", AgentBench-DB's "use the `execute_sql`
    function", AgentBench-KG's "use the tools provided".
  - We do NOT vendor or inline the tool schemas anywhere — not in the system
    prompt text, not in the user message, not via any tool-calling API
    (our pipeline is text-only; whiflash_generate has no native tool slot).

What this means for the model:

  - It sees a system prompt that says "call tool X" but has no concrete tool
    list to reference. In practice it produces tool-call-flavoured text —
    invents plausible function names, writes JSON-shaped arguments, emits
    SQL/bash inside markdown code fences — without those calls matching any
    real schema.
  - It therefore cannot actually SOLVE the underlying agentic tasks: no shell
    is executed for AgentBench-OS, no Freebase is queried for AgentBench-KG,
    no MySQL runs for AgentBench-DB, no tau-bench environment exists.

What this means for our benchmark:

  - This is FINE for measuring speculative-decoding throughput / acceptance
    length, because what matters there is the generation DISTRIBUTION.
    The model's output looks agentic even when it isn't actually completing tasks.
  - This is NOT a faithful benchmark of TASK SUCCESS. If anyone ever wants
    to report agentic accuracy numbers using these datasets, the skip-tools
    setup must be replaced with a real tool-calling harness (per-domain tool
    schema vendoring + an executor loop). That is out of scope for the spec-decoding benchmarks here.
  - Mirrors the existing SWE-bench treatment in this repo (problem statement
    only, no patch-application harness).
"""

# ==========================================================================
# tau-bench (sierra-research/tau2-bench)
# Verbatim from src/tau2/agent/llm_agent.py:
#   - AGENT_INSTRUCTION at L25–L33
#   - SYSTEM_PROMPT at L35–L41
# https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/llm_agent.py
# ==========================================================================

TAU_AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\
"""

TAU_SYSTEM_TEMPLATE = """\
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>\
"""

# ==========================================================================
# AgentBench-OS (THUDM/AgentBench)
# Verbatim system message from src/server/tasks/os_interaction/task.py L516–L524:
# https://github.com/THUDM/AgentBench/blob/main/src/server/tasks/os_interaction/task.py#L516
# User-message format from same file L530.
# ==========================================================================

OS_SYSTEM = """\
You are an assistant that will act like a person. I will play the role of a Linux (Ubuntu) operating system.
Your goal is to implement the operations required by me or answer the questions proposed by me.
For each of your turns, you should first think about what you should do, and then call exactly one of the provided tools according to the situation.
If you think the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself.
Attention, your bash code should not contain any input operation. Once again, you should use one tool in each turn, and should not respond without function calling.
Note that if you think the task has been finished, or there is some message missing to completely complete the task, you should respond with calling the function "finish_action", as no additional information will be provided.
Also, note that if you have gotten the answer to the question, you should call the "answer_action" tool instead of simply writing your answer in your response.
Your answers should be exact and precise (for example, a single number), do not answer with full sentences or phrases.
Always use a tool provided instead of simply responding with content.\
"""

# User-message format from os_interaction/task.py L530.
OS_USER_FMT = "Now, I will start a new problem in a new OS. My problem is:\n\n{description}"

# ==========================================================================
# AgentBench-KG (THUDM/AgentBench)
# Verbatim INSTRUCTIONS from src/server/tasks/knowledgegraph/const.py:
# https://github.com/THUDM/AgentBench/blob/main/src/server/tasks/knowledgegraph/const.py
# User-message format from src/server/tasks/knowledgegraph/task.py L91.
# ==========================================================================

KG_INSTRUCTIONS = """
You are an intelligent agent tasked with answering questions based on the knowledge stored in a
knowledge base (KB). You can utilize several tools to probe the KB and retrieve relevant
information to answer the questions posed to you.
Use the tools provided effectively to navigate through the KB, find relationships,
attributes, and intersections where applicable, and determine the most pertinent
information to answer the questions.
Remember:
- A variable can be an entity or a set of entities as results from previous queries.
- You should always ensure the tool used is appropriate for the question's demands and
follow the logical order where necessary (e.g., fetch relations before finding neighbors).
- After a variable is produced along the process, you need to judge whether a variable is
the final answer to the question. Each variable is represented as an id starting from 0.
For example, #0 is the first variable, #1 is the second variable, and so on.
- Once you identify the final answer to the question, respond with 'Final Answer: #id'
where id is the identifier of the variable you determine to be the final answer. For
example, if you think #3 is the final answer, you MUST respond with 'Final Answer: #3'. Do
not call tools regarding the commission of final answer!!!
- You need to execute one action at a time and can perform a maximum of {max_round} actions to find
the answer.
- You should use the supplied tools unless the final answer is identifies.
Your thoughtful application of these tools and careful consideration of the interactions
will guide you to the correct answers.
Note that you must finish the task in {max_round} rounds! Arrange your attempts according to it!
"""

# User-message format from knowledgegraph/task.py L91.
KG_USER_FMT = "Question: {question}\nEntities: [{entities}]"

# ==========================================================================
# AgentBench-DB (THUDM/AgentBench)
# Verbatim SYSTEM_PROMPT from src/server/tasks/dbbench/task.py L23–L34:
# https://github.com/THUDM/AgentBench/blob/main/src/server/tasks/dbbench/task.py
# User-message construction from same file L108–L114.
# ==========================================================================

DB_SYSTEM = """\
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer with the two specific tools provided.
If you should execute a SQL query, use the `execute_sql` function, Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then use the `commit_final_answer` function.
DO NOT use this tool unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
You should always use the tools provided to submit your answer. Be careful not to write it in the content field.
Your input will be raw MySQL response, you have to deal with it by yourself.\
"""


def db_user_msg(entry: dict) -> str:
    """Build the DB user message exactly as dbbench/task.py L108–L114 does.

    Conditional on `evidence` and `add_description` being present in the row,
    matching the official harness's behaviour. `description` (the NL question)
    is always last.
    """
    parts = []
    if entry.get("evidence"):
        parts.append(f"Evidence about the question: {entry['evidence']}")
    if entry.get("add_description"):
        parts.append(f"Additional table information about the question: {entry['add_description']}")
    parts.append(f"Question: {entry['description']}")
    return "\n".join(parts)
