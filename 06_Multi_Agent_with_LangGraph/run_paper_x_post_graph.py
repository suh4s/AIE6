# single_file_x_post_graph.py

import os
# import getpass # Remove getpass
import functools
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union, Tuple
import uuid
from pathlib import Path
import random
from termcolor import colored  # For colored printing (install with: pip install termcolor)
from dotenv import load_dotenv # Import load_dotenv
import re # Import regex for title extraction

# --- Load Environment Variables ---
# Load variables from .env file into environment variables
# Call this *before* accessing environment variables
load_dotenv()
print(".env file loaded (if found).")

# LangChain & LangGraph imports (install required packages)
# pip install langchain langgraph langchain-openai arxiv typing_extensions termcolor qdrant-client pymupdf tiktoken pyppeteer
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.tools import ArxivQueryRun
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# --- API Key Setup ---
# Now, simply check if the environment variable exists after loading .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(colored("Error: OPENAI_API_KEY not found in environment variables or .env file.", "red"))
    print(colored("Please ensure it's set in your .env file or environment.", "red"))
    exit()
# The key is implicitly used by the ChatOpenAI client when initialized without api_key argument

# --- Model Definition ---
# Using a capable model for routing and generation
try:
    # Initialize without explicitly passing the key
    llm = ChatOpenAI(model="gpt-4o-mini")
    # Optional: Add a dummy call to verify connection early
    # llm.invoke("test")
except Exception as e:
    print(colored(f"Error initializing OpenAI model: {e}", "red"))
    print(colored("Please ensure your API key is correct and you have the necessary permissions/billing setup.", "red"))
    exit()

# --- Working Directory Setup ---
WORKING_DIRECTORY = Path("./x_post_graph_output")
WORKING_DIRECTORY.mkdir(exist_ok=True)
print(f"Working directory set to: {WORKING_DIRECTORY.resolve()}")

# ==================================
# --- Tool Definitions ---
# ==================================

# Tool for fetching arXiv paper summary
try:
    arxiv_tool = ArxivQueryRun()
except Exception as e:
    print(f"Error initializing ArxivQueryRun: {e}")
    print("Ensure the 'arxiv' library is installed: pip install arxiv")
    exit()

# Simple file writing tool
@tool
def write_file(filename: str, content: str) -> str:
    """Writes the given content to a file in the working directory."""
    file_path = WORKING_DIRECTORY / filename
    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote content to {filename}"
    except Exception as e:
        return f"Error writing to {filename}: {e}"

# Simple file reading tool
@tool
def read_file(filename: str) -> str:
    """Reads the content of a file from the working directory."""
    file_path = WORKING_DIRECTORY / filename
    if not file_path.is_file():
        return f"Error: File {filename} not found in {WORKING_DIRECTORY}."
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file {filename}: {e}"

# List files tool
@tool
def list_files() -> str:
    """Lists files currently present in the working directory."""
    try:
        files = [f.name for f in WORKING_DIRECTORY.iterdir() if f.is_file()]
        if not files:
            return "No files found in the working directory."
        return "Files in working directory:\n- " + "\n- ".join(files)
    except Exception as e:
        return f"Error listing files: {e}"

# Combine file tools for teams that need them
file_tools = [write_file, read_file, list_files]

# ==================================
# --- Helper Functions (from Notebook) ---
# ==================================

# Agent Node Helper
def agent_node(state: Dict, agent: Runnable, name: str) -> Dict:
    """Runs the agent and formats output as an AIMessage."""
    # AgentExecutor returns a dict with 'output' key
    result = agent.invoke(state)
    # Ensure content is string
    content = str(result.get("output", ""))
    # The final output of an agent run is best represented as an AIMessage
    print(colored(f"--- Agent {name} Output: {content[:100]}...", "grey")) # Added print for visibility
    return {"messages": [AIMessage(content=content, name=name)]}

# Agent Creation Helper Function
def create_agent(
    llm_to_use: ChatOpenAI,
    tools_list: List[BaseTool],
    system_prompt_str: str,
) -> AgentExecutor:
    """Create a function-calling agent executor."""
    system_prompt_str += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason! Your team is: {team_members}." # Placeholder for team context
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str,),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm_to_use, tools_list, prompt)
    executor = AgentExecutor(agent=agent, tools=tools_list, verbose=False) # Set verbose=False
    return executor

# Supervisor Helper Function
def create_team_supervisor(
    llm_to_use: ChatOpenAI, system_prompt_str: str, member_names: List[str]
) -> Runnable:
    """Creates a supervisor runnable to route tasks to team members."""
    options = ["FINISH"] + member_names
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(member_names))

    return (
        prompt
        | llm_to_use.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# ==================================
# --- Research Team Definition ---
# ==================================
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    summary: str

research_tool_node = ToolNode([arxiv_tool]) # Node to run the arXiv tool

def run_arxiv_tool(state: ResearchTeamState) -> Dict[str, Any]:
    """Runs the arXiv tool directly and updates the state with the summary."""
    print(colored("--- Research Team: Fetching arXiv summary ---", "cyan"))
    query = state['query']
    summary_content = f"Error: Failed to fetch summary for query '{query}'."
    try:
        # Invoke the arxiv_tool directly as it's a Runnable
        summary_content = arxiv_tool.invoke({"query": query})
        # Basic check if the tool returned something meaningful
        if not isinstance(summary_content, str) or not summary_content.strip():
             summary_content = f"Warning: Arxiv tool returned empty or invalid content for query '{query}'."

    except Exception as e:
        print(colored(f"Error running arXiv tool directly: {e}", "red"))
        summary_content = f"Error fetching summary for query '{query}': {e}"

    # Return the summary and a message indicating completion
    return {
        "summary": summary_content,
        "messages": [HumanMessage(content=f"Fetched summary for query: {query}. Result stored in 'summary' field.")],
    }

research_graph_builder = StateGraph(ResearchTeamState)
research_graph_builder.add_node("fetch_summary", run_arxiv_tool)
research_graph_builder.set_entry_point("fetch_summary")
research_graph_builder.add_edge("fetch_summary", END)
research_graph = research_graph_builder.compile()
print("Research graph compiled.")
print("\n--- Research Graph ASCII Structure ---")
research_graph.get_graph().print_ascii()

# ==================================
# --- Authoring Team Definition ---
# ==================================
class AuthoringState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    summary: str
    draft_content: str # Not strictly needed if agents use files, but good for state tracking
    team_members: List[str]
    next: str

# Agents
x_post_drafter = create_agent(
    llm,
    file_tools,
    "You are an expert social media copywriter specializing in X (Twitter)."
    " Your task is to generate a concise and engaging post about a research paper summary."
    " The post MUST be **strictly under 280 characters**. Use relevant hashtags (1-3)."
    " Read the summary provided in the messages."
    " **If the message includes revision feedback, read the existing draft from 'x_draft.txt' using 'read_file', revise it based *only* on the feedback provided (pay close attention if feedback mentions 'Fact Check Failed'), ensuring the result is under 280 characters, and save the revised version back to 'x_draft.txt' using 'write_file'.**"
    " **Otherwise (if no revision feedback is provided), write a completely new draft based on the summary, ensuring it is under 280 characters, and save it to 'x_draft.txt' using 'write_file'.**"
    " **Verify the character count *before* saving the file.** Ensure the file is written before finishing."
)
x_post_drafter_node = functools.partial(agent_node, agent=x_post_drafter, name="X_Post_Drafter")

conciseness_editor = create_agent(
    llm,
    file_tools,
    "You are an expert editor focusing on brevity and clarity for X (Twitter) posts."
    " Use 'read_file' to read 'x_draft.txt'. Edit it to be as clear and concise as possible, "
    " ensuring it retains the core message and remains **strictly under 280 characters**."
    " **Verify the final character count *before* overwriting the file.** Use 'write_file' to overwrite 'x_draft.txt' with your final edited version. Ensure the file is overwritten before finishing."
)
conciseness_editor_node = functools.partial(agent_node, agent=conciseness_editor, name="Conciseness_Editor")

# Supervisor
authoring_members = ["X_Post_Drafter", "Conciseness_Editor"]
authoring_supervisor_agent = create_team_supervisor(
    llm,
    "You are the supervisor for the X post authoring team. Your team includes X_Post_Drafter and Conciseness_Editor."
    " Your goal is to produce a final, concise X post saved in 'x_draft.txt'."
    " 1. Start by routing to X_Post_Drafter."
    " 2. After drafting, route to Conciseness_Editor."
    " 3. After editing, FINISH.",
    authoring_members
)

# Build Graph
authoring_graph_builder = StateGraph(AuthoringState)
authoring_graph_builder.add_node("X_Post_Drafter", x_post_drafter_node)
authoring_graph_builder.add_node("Conciseness_Editor", conciseness_editor_node)
authoring_graph_builder.add_node("supervisor", authoring_supervisor_agent)

# Edges
authoring_graph_builder.add_edge("X_Post_Drafter", "supervisor")
authoring_graph_builder.add_edge("Conciseness_Editor", "supervisor")
authoring_graph_builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {member: member for member in authoring_members} | {"FINISH": END},
)
authoring_graph_builder.set_entry_point("supervisor")
authoring_graph = authoring_graph_builder.compile()
print("Authoring graph compiled.")
print("\n--- Authoring Graph ASCII Structure ---")
authoring_graph.get_graph().print_ascii()

# ==================================
# --- Verification Team Definition ---
# ==================================
class VerificationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    summary: str
    draft_content: str
    # Verification results
    verified: bool # Overall status after aggregation
    comments: str
    fact_check_status: str # "passed", "failed", "not_run"
    style_check_status: str # "passed", "failed", "not_run"
    # Control
    team_members: List[str]
    next: str

# Agents
fact_checker = create_agent(
    llm,
    [read_file, list_files], # Only needs to read
    "You are a meticulous fact-checker. Your task is to read the research summary (provided in initial message) "
    "and use 'read_file' to get the draft X post from 'x_draft.txt'. Compare them ONLY for factual accuracy. "
    "Report 'ACCURATE' if the draft correctly reflects the summary's key points, "
    "or 'INACCURATE' followed by specific discrepancies if it doesn't."
)
fact_checker_node = functools.partial(agent_node, agent=fact_checker, name="FactChecker")

x_style_checker = create_agent(
    llm,
    [read_file, list_files], # Only needs to read
    "You are an expert on X (Twitter) conventions. Use 'read_file' to read 'x_draft.txt'."
    " Check if it fits X's style: under 280 characters, engaging tone, appropriate use of 1-3 relevant hashtags. "
    " Report 'STYLE OK' if it meets the criteria, or 'STYLE ISSUE' followed by specific problems (e.g., 'Too long', 'Tone unsuitable', 'Hashtag misuse')."
)
x_style_checker_node = functools.partial(agent_node, agent=x_style_checker, name="X_StyleChecker")

# Supervisor
verification_members = ["FactChecker", "X_StyleChecker"]
verification_supervisor_prompt = (
    "You are the supervisor for the Verification Team (FactChecker, X_StyleChecker)."
    " Your goal is to verify the draft post ('x_draft.txt') against the summary (provided in messages)."
    " Follow these steps strictly:"
    " 1. Always route to FactChecker first."
    " 2. After FactChecker runs, examine its output message in the history."
    " 3. If FactChecker reported 'INACCURATE', the fact check has failed. Route immediately to FINISH."
    " 4. If FactChecker reported 'ACCURATE', the fact check passed. Route to X_StyleChecker."
    " 5. After X_StyleChecker runs, route to FINISH."
)
verification_supervisor_agent = create_team_supervisor(
    llm,
    verification_supervisor_prompt,
    verification_members
)

# Node to aggregate verification results
def aggregate_verification(state: VerificationState) -> Dict[str, Any]:
    """Aggregates feedback from verification agents, prioritizing fact check status for comments."""
    comments = []
    verified = False # Default to False unless explicitly passed
    fact_check_report = "Not run or report missing."
    style_check_report = "Not run or report missing."
    fact_check_status = "not_run"
    style_check_status = "not_run"
    fact_check_msg_found = False
    style_check_msg_found = False
    fact_check_explicitly_passed = False

    # Find the relevant messages, looking at recent ones first
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and hasattr(msg, 'name'):
            if msg.name == "FactChecker" and not fact_check_msg_found:
                fact_check_report = msg.content
                fact_check_msg_found = True
                if "ACCURATE" in msg.content.upper():
                    fact_check_explicitly_passed = True
            elif msg.name == "X_StyleChecker" and not style_check_msg_found:
                style_check_report = msg.content
                style_check_msg_found = True
            if fact_check_msg_found and style_check_msg_found:
                break # Stop if we found the most recent of both

    # Determine final status based on findings, prioritizing fact check failure
    if fact_check_msg_found:
        if fact_check_explicitly_passed:
            fact_check_status = "passed"
            # Fact check passed, now evaluate style check
            if style_check_msg_found:
                if "STYLE OK" in style_check_report.upper():
                    style_check_status = "passed"
                    verified = True # Both passed
                    comments.append("Verification Passed (Fact & Style).")
                else:
                    style_check_status = "failed"
                    # verified remains False
                    comments.append(f"Style Check Failed: {style_check_report}") # Style failure is the reason
            else:
                # Fact check passed, but style check didn't run/report
                style_check_status = "not_run"
                # verified remains False
                comments.append("Fact check passed, but style check report missing.")
        else:
            # Fact check failed (didn't contain "ACCURATE")
            fact_check_status = "failed"
            # verified remains False
            comments.append(f"Fact Check Failed: {fact_check_report}") # Fact failure is the reason
            style_check_status = "skipped_due_to_fact_fail" if style_check_msg_found else "not_run"
    else:
        # Fact check never ran or reported back
        fact_check_status = "not_run"
        # verified remains False
        comments.append("Fact check report missing.")
        style_check_status = "not_run"

    # Print results
    print(colored(f"--- Verification Result ---", "yellow"))
    print(colored(f"  Fact Check Report: {fact_check_report} (Status: {fact_check_status})", "cyan"))
    print(colored(f"  Style Check Report: {style_check_report} (Status: {style_check_status})", "cyan"))
    print(colored(f"  Overall Verified: {verified}", "cyan"))
    print(colored(f"  Final Comments: {' '.join(comments)}", "cyan"))
    print(colored(f"--------------------------", "yellow"))

    # Prepare final state update
    final_state_update = {
        "verified": verified,
        "comments": " ".join(comments) if comments else "No comments.",
        "fact_check_status": fact_check_status,
        "style_check_status": style_check_status,
        "messages": state['messages'] + [HumanMessage(content="Verification aggregation complete.")]
    }
    return final_state_update


# Build Graph
verification_graph_builder = StateGraph(VerificationState)
verification_graph_builder.add_node("FactChecker", fact_checker_node)
verification_graph_builder.add_node("X_StyleChecker", x_style_checker_node)
verification_graph_builder.add_node("supervisor", verification_supervisor_agent)
verification_graph_builder.add_node("aggregate_results", aggregate_verification)

# Edges: Supervisor directs flow based on its internal logic and state
verification_graph_builder.add_edge("FactChecker", "supervisor") # FactChecker reports back
verification_graph_builder.add_edge("X_StyleChecker", "supervisor") # StyleChecker reports back

# Conditional edges from the supervisor
verification_graph_builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next"], # Supervisor decides next step ('FactChecker', 'X_StyleChecker', or 'FINISH')
    {
        "FactChecker": "FactChecker",
        "X_StyleChecker": "X_StyleChecker",
        "FINISH": "aggregate_results", # Supervisor routes to FINISH which triggers aggregation
    }
)
verification_graph_builder.add_edge("aggregate_results", END) # End after aggregation
verification_graph_builder.set_entry_point("supervisor") # Supervisor starts the verification process
verification_graph = verification_graph_builder.compile()
print("Verification graph compiled.")
print("\n--- Verification Graph ASCII Structure ---")
verification_graph.get_graph().print_ascii()


# ==================================
# --- Meta-Supervisor & Full Graph ---
# ==================================

MAX_REVISIONS = 2 # Set a limit for revisions

class OverallState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # Paper Info
    query: str
    summary: str
    paper_title: Optional[str] # Added
    # Draft Info
    draft_content: str
    # Verification Info
    verified: bool
    comments: str
    # Control Flow & Logging
    revision_count: int
    next: str # For meta-supervisor routing
    execution_log: Annotated[List[Tuple[int, str, str]], operator.add] # Added for logging (step, node, details)

# --- Enhanced Subgraph Runner Functions ---

def run_research(state: OverallState) -> Dict[str, Any]:
    """Invokes the research graph, initializes state, logs execution."""
    step_num = len(state.get("execution_log", [])) + 1
    query = state["query"]
    print(colored(f"\n[{step_num}] === Running Research Team ===", "green", attrs=["bold"]))
    print(colored(f"  Task: Fetch summary for query: '{query}'", "green"))
    print(colored(f"  Typical Next Step: Supervisor -> Authoring", "green"))

    research_input = {"query": query, "messages": []}
    research_result = research_graph.invoke(research_input)
    summary = research_result.get('summary', "Error: Summary not found in research result.")

    # Attempt to extract title (simple regex, might need refinement)
    paper_title = f"Paper on '{query}'" # Default title
    if isinstance(summary, str):
        match = re.search(r"Title:\\s*(.*)", summary, re.IGNORECASE)
        if match:
            paper_title = match.group(1).strip()
    print(colored(f"  Paper Title Found: {paper_title}", "magenta"))
    print(colored(f"  Summary Fetched (first 100 chars): {summary[:100]}...", "magenta"))

    log_entry = (step_num, "Research", f"Fetch summary for '{query}'")

    # Initialize/reset state fields for the new workflow run
    return {
        "summary": summary,
        "paper_title": paper_title,
        "revision_count": 0,
        "verified": False, # Reset verification status
        "comments": "",    # Reset comments
        "execution_log": [log_entry], # Start the log
        "messages": [HumanMessage(content="Research complete.")]
    }

def run_authoring(state: OverallState) -> Dict[str, Any]:
    """Invokes the authoring graph, handles revisions, logs execution."""
    step_num = len(state.get("execution_log", [])) + 1
    paper_context = state.get('paper_title') or state.get('query', 'Unknown Paper')
    print(colored(f"\n[{step_num}] === Running Authoring Team ===", "green", attrs=["bold"]))
    print(colored(f"  Context: Processing '{paper_context}'", "green"))
    print(colored(f"  Typical Source: Supervisor decision", "green"))
    print(colored(f"  Typical Next Step: Supervisor -> Verification", "green"))

    summary = state.get("summary")
    log_detail = "Skipped: Invalid summary"
    if not summary or "Error" in summary:
        print(colored("Skipping Authoring: Summary is missing or contains an error.", "red"))
        return {"messages": [HumanMessage(content="Skipping Authoring: Invalid summary.")], "execution_log": [(step_num, "Authoring", log_detail)]}

    comments = state.get("comments", "")
    revision_requested = bool(comments)
    revision_count = state.get("revision_count", 0)

    if revision_requested:
        log_detail = f"Revise draft (Attempt {revision_count + 1})"
        print(colored(f"  Task: {log_detail} based on comments: {comments[:100]}...", "yellow"))
        initial_message_content = f"""Please revise the draft in 'x_draft.txt' based on the following feedback:
{comments}

Original Summary was:
{summary}
"""
        next_revision_count = revision_count + 1
    else:
        log_detail = "Create initial draft"
        print(colored(f"  Task: {log_detail}", "yellow"))
        initial_message_content = f"""Draft an X post based on this summary:

{summary}
"""
        next_revision_count = revision_count

    initial_message = HumanMessage(content=initial_message_content)
    authoring_input = {"summary": summary, "messages": [initial_message]}
    authoring_result = authoring_graph.invoke(authoring_input)

    final_draft = read_file.invoke({"filename": "x_draft.txt"})
    print(colored(f"  Draft after authoring step: {final_draft}", "magenta"))

    log_entry = (step_num, "Authoring", log_detail)

    return {
        "draft_content": final_draft,
        "revision_count": next_revision_count,
        "comments": "", # Clear old comments
        "execution_log": [log_entry], # Append log
        "messages": [HumanMessage(content="Authoring complete.")]
    }

def run_verification(state: OverallState) -> Dict[str, Any]:
    """Invokes the verification graph, logs execution."""
    step_num = len(state.get("execution_log", [])) + 1
    paper_context = state.get('paper_title') or state.get('query', 'Unknown Paper')
    print(colored(f"\n[{step_num}] === Running Verification Team ===", "green", attrs=["bold"]))
    print(colored(f"  Context: Verifying draft for '{paper_context}'", "green"))
    print(colored(f"  Typical Source: Supervisor decision", "green"))
    print(colored(f"  Typical Next Step: Supervisor (makes decision)", "green"))

    summary = state.get("summary")
    draft_to_verify = read_file.invoke({"filename": "x_draft.txt"})
    log_detail = "Checking draft"

    if "Error: File" in draft_to_verify:
         log_detail = "Skipped: Draft file missing"
         print(colored(f"{log_detail}.", "red"))
         return {
             "messages": [HumanMessage(content="Skipping Verification: Draft file missing.")],
             "verified": False, "comments": "Draft file missing.",
             "execution_log": [(step_num, "Verification", log_detail)]
         }
    if not summary or "Error" in summary:
         log_detail = "Skipped: Summary missing/invalid"
         print(colored(f"{log_detail}.", "red"))
         return {
             "messages": [HumanMessage(content="Skipping Verification: Invalid summary.")],
             "verified": False, "comments": "Summary missing or invalid.",
             "execution_log": [(step_num, "Verification", log_detail)]
         }

    print(colored(f"  Task: {log_detail}", "yellow"))
    initial_message = HumanMessage(
        content=f"Verify the draft post ('x_draft.txt') against the summary:\\n\\nSummary:\\n{summary}"
    )
    verification_input = { "summary": summary, "draft_content": draft_to_verify, "messages": [initial_message] }
    verification_result = verification_graph.invoke(verification_input)

    verified_status = verification_result.get("verified", False)
    verification_comments = verification_result.get("comments", "Verification comments missing.")
    log_detail = f"Result: Verified={verified_status}. Comments: {verification_comments[:50]}..."

    log_entry = (step_num, "Verification", log_detail)

    # Return results for the supervisor
    return {
        "verified": verified_status,
        "comments": verification_comments,
        "execution_log": [log_entry], # Append log
        "messages": [HumanMessage(content="Verification complete.")],
    }

# --- Define the Meta-Supervisor Agent (Using detailed prompt from previous step) ---
all_teams = ["Research", "Authoring", "Verification"]
meta_supervisor_system_prompt = (
    f"You are the Meta-Supervisor overseeing Research, Authoring, and Verification teams for creating an X post about a research paper."
    f" Your goal is to manage the workflow to produce a verified post. The maximum number of revision attempts is {MAX_REVISIONS}."
    f" Based on the conversation history and the current state (especially 'verified' status, 'comments', and 'revision_count'), decide the next team to act."
    f" Workflow Steps & Routing Logic:"
    f" 1. Start with the Research team." # Implicit via entry point
    f" 2. After Research, always proceed to Authoring."
    f" 3. After Authoring, always proceed to Verification."
    f" 4. After Verification: Examine the 'verified' status and 'comments' from the verification result."
    f"    a. If 'verified' is True, the process is complete, route to FINISH."
    f"    b. If 'verified' is False, check the 'revision_count'."
    f"    c. If 'revision_count' < {MAX_REVISIONS}, route back to Authoring for revision. Ensure the 'comments' field contains the feedback (prioritizing fact-check failures)."
    f"    d. If 'revision_count' >= {MAX_REVISIONS}, the post could not be verified within the allowed attempts, route to FINISH."
    f" Select the next team from {all_teams} or FINISH."
)
meta_supervisor_agent = create_team_supervisor(llm, meta_supervisor_system_prompt, all_teams)

# --- Wrapper for Supervisor Node to Add Logging ---
def supervisor_node_wrapper(state: OverallState) -> Dict[str, Any]:
    """Runs the supervisor agent and logs its decision."""
    step_num = len(state.get("execution_log", [])) + 1
    paper_context = state.get('paper_title') or state.get('query', 'Unknown Paper')
    print(colored(f"\n[{step_num}] === Running Supervisor ===", "blue", attrs=["bold"]))
    print(colored(f"  Context: Deciding next step for '{paper_context}'", "blue"))
    print(colored(f"  Current State: Verified={state.get('verified')}, Rev Count={state.get('revision_count')}, Comments='{state.get('comments', '')[:50]}...'", "blue"))

    # Prepare state for supervisor (it mainly needs messages and state vars mentioned in prompt)
    # Filter state to only include keys expected by the supervisor's context/logic
    supervisor_input_state = {
        "messages": state.get("messages", []),
        "verified": state.get("verified", False), # Pass verification status
        "comments": state.get("comments", ""),    # Pass comments
        "revision_count": state.get("revision_count", 0) # Pass revision count
    }
    # Add other fields IF they are explicitly used in the supervisor prompt (e.g. 'summary')
    # supervisor_input_state["summary"] = state.get("summary","")

    result = meta_supervisor_agent.invoke(supervisor_input_state) # Invoke the actual supervisor

    next_node = result.get("next", "ERROR")
    if next_node == "ERROR":
        print(colored("  Supervisor ERROR: Failed to determine next step.", "red"))
        # Default to FINISH on error to avoid loops? Or raise exception?
        next_node = "FINISH" # Decide error handling strategy
    print(colored(f"  Supervisor Decision: Route to -> {next_node}", "yellow"))

    log_entry = (step_num, "Supervisor", f"Decision: Route to {next_node}")

    # Return the original result + the log entry + necessary state (supervisor doesn't modify state directly)
    return {"next": next_node, "execution_log": [log_entry]}


# --- Build the Final Graph (Using Supervisor Wrapper) ---
final_graph_builder = StateGraph(OverallState)

final_graph_builder.add_node("Research", run_research)
final_graph_builder.add_node("Authoring", run_authoring)
final_graph_builder.add_node("Verification", run_verification)
final_graph_builder.add_node("Supervisor", supervisor_node_wrapper) # Use the wrapper

# --- Define Edges for Final Graph ---
final_graph_builder.add_edge("Research", "Supervisor")
final_graph_builder.add_edge("Authoring", "Supervisor")
final_graph_builder.add_edge("Verification", "Supervisor")

# Conditional edges *from the Supervisor Wrapper*
final_graph_builder.add_conditional_edges(
    "Supervisor",
    # Route based on the 'next' value returned by the wrapper
    lambda x: x["next"],
    {team: team for team in all_teams} | {"FINISH": END},
)

# Start with Research, then let supervisor wrapper handle logging and routing
final_graph_builder.set_entry_point("Research")
final_graph = final_graph_builder.compile()
print("Final graph compiled with LLM supervisor and revision loop logic.")
print("\n--- Final Graph ASCII Structure ---")
final_graph.get_graph().print_ascii()

# === Pause before execution ===
input(colored("\nGraph structures printed. Press Enter to start execution...", "magenta"))


# ==================================
# --- Mermaid Diagram Generation ---
# ==================================
def generate_execution_mermaid(log: List[Tuple[int, str, str]]) -> str:
    """Generates a Mermaid Markdown diagram from the execution log."""
    if not log:
        return "No execution log found to generate diagram."

    mermaid = ["```mermaid", "graph TD"]
    nodes = set()
    edge_labels = {} # To store labels for edges (step number)

    # Define nodes based on log
    all_node_names = ["Start"] + [name for _, name, _ in log] + ["End"]
    for name in sorted(list(set(all_node_names))):
        # Use different shapes for clarity
        shape_open, shape_close = "([", "])" # Default: Stadium for Start/End
        if name in ["Research", "Authoring", "Verification"]:
            shape_open, shape_close = "[", "]" # Box for team runners
        elif name == "Supervisor":
            shape_open, shape_close = "{(", ")}" # Diamond for supervisor decision point
        mermaid.append(f"    {name}{shape_open}\"{name}\"{shape_close}")
        nodes.add(name)

    # Create edges and aggregate step numbers
    last_node = "Start"
    for step, node_name, details in log:
        edge = (last_node, node_name)
        if edge not in edge_labels:
            edge_labels[edge] = []
        edge_labels[edge].append(str(step))
        last_node = node_name

    # Add final edge to End (if log is not empty)
    if log:
        edge = (last_node, "End")
        if edge not in edge_labels:
            edge_labels[edge] = []
        edge_labels[edge].append(str(len(log) + 1)) # Number the final step

    # Add edges with aggregated step numbers to Mermaid syntax
    for (u, v), steps in edge_labels.items():
        label = ", ".join(steps)
        mermaid.append(f"    {u} -- Step(s) {label} --> {v}")


    # Basic Styling (optional - using Mermaid default is often fine)
    mermaid.append("\n    %% Optional Styling")
    mermaid.append("    classDef default fill:#eee,stroke:#333,stroke-width:1px;")
    mermaid.append("```")

    return "\n".join(mermaid)

# ==================================
# --- Social Media Post Generation ---
# ==================================
def generate_social_posts(state: OverallState):
    """Generates formatted posts for X and Discord based on the final state."""
    print("\n" + colored("==================================", "yellow", attrs=["bold"]))
    print(colored("Generating Social Media Posts...", "yellow", attrs=["bold"]))
    print(colored("==================================", "yellow", attrs=["bold"]))

    query = state.get("query", "N/A")
    paper_title = state.get("paper_title", f"Paper on '{query}'")
    summary = state.get("summary", "Summary not available.")
    verified = state.get("verified", False)
    comments = state.get("comments", "")
    arxiv_link = f"https://arxiv.org/abs/{query}" if query != "N/A" and not query.startswith("Error") else "(Link not available)"
    draft_content = read_file.invoke({"filename": "x_draft.txt"})
    if "Error: File" in draft_content:
        print(colored("Could not read final draft file for post generation.", "red"))
        draft_content = "[Draft generation failed]"
        verified = False

    # --- Generate X Post --- 
    print(colored("\n--- X (Twitter) Post Draft(s) ---", "cyan"))
    x_posts_list = []
    base_hashtags = f"#AI #LLM #{paper_title.split()[0].replace(':','')} #LangGraph"
    ai_makerspace_shoutout = "\n\nLearned so much building this with LangGraph in the @AIMakerspace AI Engineering Bootcamp! Highly recommend it. 🔥"

    if verified:
        # Define the intro explaining the multi-agent process
        intro_message = (
            f"🤖✨ Built a multi-agent app with #LangGraph (@AIMakerspace Bootcamp!) to summarize ML papers! "
            f"It fetches from arXiv, drafts a post, & verifies it. This time: '{paper_title}' ({arxiv_link}) yielded this summary:"
        )
        # Use the verified draft content directly
        summary_content = draft_content

        # Combine intro + summary for potential single post check
        full_message_no_extras = intro_message + "\n\n" + summary_content

        # Calculate available chars for the *content* part
        approx_chars_limit = 280
        shoutout_len = len(ai_makerspace_shoutout)
        hashtags_len = len(base_hashtags)
        # Buffer for potential numbering (e.g., " (1/2)"), spacing, etc.
        buffer = 15
        available_chars_for_content = approx_chars_limit - buffer - shoutout_len - hashtags_len

        if len(full_message_no_extras) <= available_chars_for_content:
            # Fits in one post
            print(colored("(Content fits in single post)", "yellow"))
            single_post = full_message_no_extras + ai_makerspace_shoutout + "\n" + base_hashtags
            x_posts_list.append(single_post)
        else:
            # Needs threading
            print(colored("(Content too long for single post, attempting to thread...)", "yellow"))

            # Start first post with the intro
            first_post_text = intro_message
            remaining_content_to_thread = summary_content
            temp_posts = []

            # Add first post (intro only, might add some summary if space)
            # Calculate space left in first post after intro
            available_chars_first = approx_chars_limit - len(first_post_text) - 10 # buffer for (1/n)
            
            # Try splitting remaining content by sentences
            import textwrap
            sentences = [s.strip() for s in remaining_content_to_thread.split('.') if s.strip()]
            if not sentences and remaining_content_to_thread: # Handle cases with no periods
                sentences = [remaining_content_to_thread.strip()]

            current_thread_part = "" # Content for subsequent posts
            first_post_finished = False

            # Add sentences to first post if they fit
            first_post_added_content = False
            processed_sentence_count = 0
            for i, sentence in enumerate(sentences):
                sentence_part = ("\n" if not first_post_added_content else " ") + sentence + "."
                if len(first_post_text) + len(sentence_part) <= available_chars_first:
                    first_post_text += sentence_part
                    first_post_added_content = True
                    processed_sentence_count += 1
                else:
                    # First post is full
                    break 
            
            temp_posts.append(first_post_text.strip())
            remaining_sentences = sentences[processed_sentence_count:]

            # Process remaining sentences for subsequent posts
            current_thread_part = ""
            for sentence in remaining_sentences:
                sentence_part = sentence + ". "
                # Check if adding the next sentence exceeds the limit (leaving space for numbering)
                if len(current_thread_part) + len(sentence_part) + 10 < approx_chars_limit:
                    current_thread_part += sentence_part
                else:
                    # Finish previous post (if exists)
                    if current_thread_part:
                        temp_posts.append(current_thread_part.strip())
                    # Start new post
                    current_thread_part = sentence_part

            # Add the last part
            if current_thread_part:
                 temp_posts.append(current_thread_part.strip())

            # Add numbering, shoutout, and hashtags to the collected posts
            total_posts = len(temp_posts)
            for i, post_text in enumerate(temp_posts):
                # Ensure numbering is added correctly
                post_with_numbering = f"{post_text} ({i+1}/{total_posts})"
                # Check length *after* adding numbering
                if len(post_with_numbering) > approx_chars_limit:
                     print(colored(f"(Warning: Post {i+1} might slightly exceed 280 chars after numbering, check manually)", "yellow"))
                
                final_post = post_with_numbering
                if i == total_posts - 1: # Add shoutout and hashtags to the last post
                    final_post += ai_makerspace_shoutout + "\n" + base_hashtags
                x_posts_list.append(final_post)

    else: # Not verified - generate fallback (already includes context)
        fallback_post = (
            f"🤖 Built a multi-agent app with #LangGraph (@AIMakerspace Bootcamp rocks!) to summarize ML papers like '{paper_title}' ({arxiv_link}). "
            f"Drafting/Verification is tricky! This one failed (Status: {verified}, Reason: {comments[:50]}...). Still learning! 🤔 #AI"
        )
        x_posts_list.append(fallback_post)

    # Print the generated post(s)
    if len(x_posts_list) == 1:
        print(x_posts_list[0])
    else:
        print(colored(f"--- Generated X Thread ({len(x_posts_list)} posts) --- ", "yellow"))
        for i, post in enumerate(x_posts_list):
            print(f"\n--- Post {i+1}/{len(x_posts_list)} ---")
            print(post)
        print(colored("------------------------------------", "yellow"))
    
    # Add instructions for posting
    print(colored("\n--- How to Post the Thread on X.com ---", "magenta"))
    print("1. Go to X.com or open the X app.")
    print("2. Click the 'Post' (or feather/plus icon) button.")
    print("3. Paste the text for 'Post 1/n' into the composer.")
    print("4. Click the '+' button below the first post in the composer to add the next post in the thread.")
    print("5. Paste the text for 'Post 2/n' into the new composer window that appears.")
    print("6. Repeat steps 4 & 5 for all remaining posts in the thread.")
    print("7. Once all posts are added, click 'Post all'.")
    print("(Note: Ensure links render correctly and consider adding relevant images/GIFs manually)")

    # --- Generate Discord Post (Keep as before) --- 
    print(colored("\n--- Discord Post Draft ---", "cyan"))
    key_takeaway = summary.split('.')[0] + '.' if '.' in summary else summary[:100] + "..."
    discord_post = (
        f"🚀 **Exciting News & Paper Summary!** 🚀\n\n"
        f"Hey #AI folks! I built a fun multi-agent application using **LangGraph** in the @AIMakerspace Bootcamp! 🎉🤖 It automatically fetches an ML paper from arXiv, drafts a summary post, and even *tries* to verify it.\n\n"
        f"This time, it processed: **{paper_title}**\n"
        f"🔗 Link: {arxiv_link}\n\n"
        f"🤔 **Quick Takeaway:** {key_takeaway}\n\n"
        f"🛠️ **The Process:**\n1. **Research Agent:** Fetched the paper summary.\n2. **Authoring Agent:** Drafted an X-style post.\n3. **Verification Agent:** Checked facts & style (Status: {'✅ Verified' if verified else f'❌ Failed ({comments})'}).\n\n"
        f"It's amazing what you can orchestrate with LangGraph! Definitely learned a lot about agent workflows (and LLM fact-checking challenges!). 🔥\n\n"
        f"#LangChain #LangGraph #MultiAgent #AI #MachineLearning #LLM #Python"
    )
    print(discord_post)


# ==================================
# --- Main Execution Block (Update) ---
# ==================================
if __name__ == "__main__":
    print(colored("\nStarting X Post Generation Process", "yellow", attrs=["bold"]))
    print(colored("==================================", "yellow", attrs=["bold"]))
    arxiv_query = "2404.19553"
    print(f"Target Paper Query: {arxiv_query}")
    print(f"\nClearing files in working directory: {WORKING_DIRECTORY}")
    for f in WORKING_DIRECTORY.glob("*.txt"):
        try:
            f.unlink()
        except OSError as e:
            print(f"Error removing file {f}: {e}")
    print("Workspace cleared.")


    # Initial State - include empty execution_log
    initial_state = {
        "query": arxiv_query,
        "messages": [HumanMessage(content=f"Create and verify an X post about arXiv paper: {arxiv_query}")],
        "execution_log": [] # Initialize log
    }

    print("\nInvoking the final graph...")
    final_state = None
    try:
        # Invoke normally - state is managed internally by LangGraph
        final_state = final_graph.invoke(initial_state, {"recursion_limit": 20}) # Increased limit slightly
    except Exception as e:
         print(colored(f"\nGRAPH EXECUTION FAILED: {e}", "red", attrs=["bold"]))
         import traceback
         traceback.print_exc()

    print(colored("\n==================================", "yellow", attrs=["bold"]))
    print(colored("Process Finished!", "green", attrs=["bold"]))

    # --- Display Results ---
    if final_state:
        print("\n--- Final State ---")
        # Optionally print parts of the final state for debugging
        # print(f"  Query: {final_state.get('query')}")
        # print(f"  Summary (start): {final_state.get('summary', 'N/A')[:100]}...")
        # print(f"  Draft Content: {final_state.get('draft_content', 'N/A')}")
        print(f"  Verified: {final_state.get('verified', 'N/A')}")
        print(f"  Comments: {final_state.get('comments', 'N/A')}")
        # print(f"  Last Message: {final_state.get('messages', [])[-1].content if final_state.get('messages') else 'N/A'}")

        final_draft_content = read_file.invoke({"filename": "x_draft.txt"})
        print(f"\n--- Final Draft (from x_draft.txt) ---")
        if "Error: File" in final_draft_content:
            print(colored(final_draft_content, "red"))
        else:
            print(colored(final_draft_content, "yellow"))

        if final_state.get('verified') is True:
            print(colored("\nPost was successfully verified!", "green"))
        else:
            print(colored(f"\nPost verification failed or was inconclusive. Comments: {final_state.get('comments', 'N/A')}", "red"))

        # --- Generate and Print Mermaid Diagram ---
        print("\n--- Execution Flow Diagram (Mermaid Markdown) ---")
        execution_log = final_state.get("execution_log", [])
        mermaid_diagram = generate_execution_mermaid(execution_log)
        print(mermaid_diagram)
        
        # --- Generate and Print Social Media Posts ---
        generate_social_posts(final_state)

    else:
        print(colored("\nCould not retrieve final state due to execution error.", "red"))

    print(colored("==================================", "yellow", attrs=["bold"]))
