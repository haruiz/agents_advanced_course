import asyncio
import json
import os
import logging
from typing import Optional, Any

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, BaseSessionService, Session
from google.adk.sessions.base_session_service import ListSessionsResponse, GetSessionConfig
from google.genai import types
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Import the agent from the main file
from submit_requests import root_agent

load_dotenv()

APPROVAL_REQUESTS_FILE = "approval_requests.json"
APP_NAME = os.getenv("APP_NAME", "memory_agent_app")
db_url = "sqlite+aiosqlite:///sessions.db"
session_service = DatabaseSessionService(db_url=db_url)
console = Console()


async def approve_requests():
    if not os.path.exists(APPROVAL_REQUESTS_FILE):
        console.print("[yellow]No pending approval requests found.[/yellow]")
        return

    try:
        with open(APPROVAL_REQUESTS_FILE, "r") as f:
            requests = json.load(f)
    except (json.JSONDecodeError, IOError):
        console.print("[red]Error reading approval requests file.[/red]")
        return

    if not requests:
        console.print("[yellow]No pending approval requests.[/yellow]")
        return

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    remaining_requests = []
    
    for req in requests:
        console.print(f"\n[bold cyan]Request ID:[/bold cyan] {req['id']}")
        console.print(f"[bold cyan]Function:[/bold cyan] {req['name']}")
        console.print(f"[bold cyan]Arguments:[/bold cyan] {req['args']}")
        console.print(f"[bold cyan]User ID:[/bold cyan] {req['user_id']}")
        
        choice = input("Approve? (y/n/skip): ").lower()
        
        if choice in ['y', 'n']:
            status = "approved" if choice == 'y' else "rejected"
            approval_response_data = {
                "status": status,
                "message": f"Manager has {status} the request.",
                "observations": "You are expending so much money!!!"
            }
            
            # Create the tool output for resumption
            approval_response_part = types.Part(
                function_response=types.FunctionResponse(
                    id=req["id"],
                    name=req["name"],
                    response=approval_response_data
                )
            )

            console.print(f"[green]Resuming session {req['session_id']} with {status}...[/green]")

            try:
                content = types.Content(role="user", parts=[approval_response_part])
                async for event in runner.run_async(
                        user_id=req["user_id"],
                        session_id=req["session_id"],
                        new_message=content
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        continue
            except Exception as e:
                console.print(f"[red]Error resuming session: {e}[/red]")
        else:
            console.print("[yellow]Skipping request.[/yellow]")
            remaining_requests.append(req)

    # Update the file with remaining requests
    with open(APPROVAL_REQUESTS_FILE, "w") as f:
        json.dump(remaining_requests, f, indent=2)

if __name__ == "__main__":
    # Note: We need to make sure async_human_in_loop is importable.
    # If the filename has hyphens, we might need to rename it or use importlib.
    # Renaming async_human-in-loop.py to async_human_in_loop.py is recommended.
    asyncio.run(approve_requests())
