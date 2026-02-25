import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, BaseSessionService
from google.adk.tools import ToolContext, LongRunningFunctionTool
from google.genai import types
# setup rich print for better logging
from rich.logging import RichHandler
from google.adk.agents.callback_context import CallbackContext

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3-pro-preview")
APPROVAL_REQUESTS_FILE = "approval_requests.json"

# Using SQLite database for persistent storage
db_url = "sqlite+aiosqlite:///sessions.db"
session_service: BaseSessionService = DatabaseSessionService(db_url=db_url)

initial_state = {}


def reimburse(purpose: str, amount: float) -> dict[str, Any]:
    """Reimburse the amount of money to the employee."""
    return {
        'status': 'ok',
        'amount': amount,
    }


def ask_for_approval(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
    """Ask for approval for the reimbursement."""
    return {
        'status': 'pending',
        'amount': amount,
        'ticketId': f'reimbursement-ticket-{str(uuid.uuid4())[:8]}',
    }


async def on_after_agent_call(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Invoked before each agent call. Used to inspect or modify context.
    """
    session_in_ctx = callback_context._invocation_context.session
    session_in_service = (
        await callback_context._invocation_context.session_service.get_session(
            app_name=session_in_ctx.app_name,
            user_id=session_in_ctx.user_id,
            session_id=session_in_ctx.id,
        )
    )
    assert session_in_service is not None
    logger.debug(f"Session context {session_in_service}")
    events = session_in_service.events
    if events:
        for event in events:
            if event.long_running_tool_ids:
                logger.info(
                    f"Long Running Tool Call Detected in session {session_in_service.id}: "
                    f"Event ID {event.id} with long running tool IDs {event.long_running_tool_ids}"
                )


root_agent = Agent(
    model=DEFAULT_MODEL,
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement.

      If the amount is greater than $100, you will
      ask for approval from the manager. If the manager approves, you will
      call reimburse() to reimburse the amount to the employee. If the manager
      rejects, you will inform the employee of the rejection.
    """,
    tools=[
        reimburse,
        LongRunningFunctionTool(func=ask_for_approval)],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
    after_agent_callback=on_after_agent_call,
)


def save_approval_request(request_data: dict):
    """Saves a pending tool call to a JSON file as a list."""
    requests = []
    if os.path.exists(APPROVAL_REQUESTS_FILE):
        try:
            with open(APPROVAL_REQUESTS_FILE, "r") as f:
                requests = json.load(f)
        except (json.JSONDecodeError, IOError):
            requests = []

    # Avoid duplicate entries for the same tool call ID
    if not any(r.get("id") == request_data["id"] for r in requests):
        requests.append(request_data)
        with open(APPROVAL_REQUESTS_FILE, "w") as f:
            json.dump(requests, f, indent=2)
        logger.info(f"Saved approval request to {APPROVAL_REQUESTS_FILE}")


async def process_event(event, user_id: str, session_id: str):
    """Process the agent's response and print it to the console."""
    if event and event.content and event.content.parts:
        for part in event.content.parts:
            if part.function_call:
                if event.long_running_tool_ids and part.function_call.id in event.long_running_tool_ids:
                    logger.info(f"Long Running Tool Call: {part.function_call.name}({part.function_call.args})")
                    
                    save_approval_request({
                        "id": part.function_call.id,
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                        "user_id": user_id,
                        "session_id": session_id,
                        "status": "pending"
                    })

    if event.is_final_response() and event.content and event.content.parts:
        for part in event.content.parts:
            if part.text:
                return part.text
    return None


async def call_agent(runner: Runner, user_id: str, session_id: str, query: str) -> None:
    """Function to call the agent asynchronously."""
    content = types.Content(role="user", parts=[types.Part(text=query)])
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            response = await process_event(event, user_id, session_id)
            if response:
                logger.info(f"Agent: {response}")
    except Exception as e:
        logger.error(f"Error during agent call: {e}")


async def main(force_new_session: bool = False):
    # Setup constants
    APP_NAME = os.getenv("APP_NAME", "memory_agent_app")
    USER_ID = os.getenv("USER_ID", "haruiz")

    # Session management
    if force_new_session:
        existing = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
        for s in existing.sessions:
            await session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=s.id)
        logger.info(f"Cleared existing sessions for user {USER_ID}")

    existing = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    if existing.sessions:
        session_id = existing.sessions[0].id
        logger.info(f"Resuming session: {session_id}")
    else:
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, state=initial_state)
        session_id = session.id
        logger.info(f"Started new session: {session_id}")

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print("\n--- Reimbursement Agent ---")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            # Use run_in_executor to prevent blocking the event loop
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            await call_agent(runner, USER_ID, session_id, user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    asyncio.run(main(force_new_session=False))
