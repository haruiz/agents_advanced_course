import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk import Runner, Agent
from google.adk.agents import LlmAgent, LoopAgent, BaseAgent, InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types
from rich.logging import RichHandler

from tools import get_db_schema, execute_sql_query
from pydantic import BaseModel

class SqlFeedback(BaseModel):
    is_valid: bool
    feedback: str
    results: str = None

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)
logger = logging.getLogger(__name__)

# Use this path for testing your agent using the Web UI since the command is run from the root folder
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", None)
DEFAULT_MODEL=os.getenv("DEFAULT_MODEL", "gemini-3-pro-preview")



def on_before_agent_call(callback_context: CallbackContext) -> None:
    """
    Invoked before each agent call. Used to inspect or modify context.

    Args:
        callback_context (CallbackContext): Execution context.
    """
    state = callback_context.state
    if "db_schema" not in state:
        schema_result = get_db_schema()
        if schema_result["status"] == "success":
            state["db_schema"] = schema_result["schema"]
        else:
            state["db_schema"] = "Error fetching schema: " + schema_result.get("error_message", "Unknown error")
    logger.debug("Before agent callback triggered. State: %s", callback_context.state)


async def on_after_agent_callback(callback_context: CallbackContext) -> None:
    """
    Invoked after each agent call. Used to inspect or modify context.

    Args:
        callback_context (CallbackContext): Execution context.
    """
    logger.debug("After agent callback triggered. State: %s", callback_context.state)

# --- Define Agents ---
sql_junior_writer_agent = LlmAgent(
    name="sql_junior_writer_agent",
    model=DEFAULT_MODEL,
    description="Generates SQL queries from user prompts.",
    global_instruction=(
        "You are a data assistant specializing in SQL. "
        f"Use this schema to write syntactically correct SQL queries"
    ),
    instruction=(
        "Use the provided schema in session state under 'db_schema' to write a SQL query. "
        "Store your result in 'sql_query'. Do not execute it."
    ),
    output_key="sql_query",
    tools=[get_db_schema],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)

sql_senior_writer_agent = Agent(
    name="sql_senior_writer_agent",
    model=DEFAULT_MODEL,
    description="Validates and executes SQL queries.",
    instruction=(
        "Review the SQL query in session state under 'sql_query' using the schema in session state under 'db_schema'. "
        "Validate it for correctness and best practices. "
        "If valid, execute it using 'execute_sql_query' and return the result."
        "The final output should be a JSON object with the following format: "
        "{ 'is_valid': bool, 'feedback': str, 'results': tabular results in markdown format if valid else None }"
    ),
    tools=[execute_sql_query, get_db_schema],
    output_schema=SqlFeedback,
    output_key="sql_results",
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)


class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        is_valid = ctx.session.state.get("sql_results", {}).get("is_valid", False)
        should_stop = not is_valid
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))



# --- Loop Agent ---
root_agent = LoopAgent(
    name="root_agent",
    sub_agents=[sql_junior_writer_agent, sql_senior_writer_agent, CheckStatusAndEscalate(name="check_status_and_escalate")],
    max_iterations=1,
    before_agent_callback=on_before_agent_call,
    after_agent_callback=on_after_agent_callback,
    description="Coordinates SQL generation and execution pipeline.",
)

# --- Runtime Entrypoint ---
async def call_agent(prompt: str) -> None:
    """
    Call the root agent with a prompt and print the final output.

    Args:
        prompt (str): Natural language query for database.
    """

    APP_NAME = "sql_assistant_agent"
    USER_ID = "dev_user_01"
    SESSION_ID = "dev_user_session_01"

    # --- Session & Runner Setup ---
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response() and event.content:
            logger.info("\n\n[Final Response]\n%s", event.content.parts[0].text)


if __name__ == "__main__":
    try:
        prompt = (
            "List the top 10 customers by total order amount"
        )
        logger.info("Calling agent with prompt: %s", prompt)
        import asyncio
        asyncio.run(call_agent(prompt))
    except Exception as e:
        logger.exception("An error occurred while running the agent.")
        print(f"Error: {e}")