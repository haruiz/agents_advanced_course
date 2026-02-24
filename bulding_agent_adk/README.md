# DB Agent Application

This application demonstrates a database agent that can generate and execute SQL queries based on natural language input. It uses the Google ADK (Agent Development Kit) to create a multi-agent system for interacting with a SQLite database.

## Overview

The application consists of the following main components:

-   **Agents**:
    -   `sql_junior_writer_agent`: An LLM-based agent that generates SQL queries from user prompts.
    -   `sql_senior_writer_agent`: An agent that validates and executes SQL queries.
    -   `root_agent`: A loop agent that coordinates the SQL generation and execution pipeline.
-   **Tools**:
    -   `get_db_schema`: A function that fetches the SQL schema of the database.
    -   `execute_sql_query`: A function that executes a SQL query and returns the results.
-   **Configuration**:
    -   Environment variables loaded from a `.env` file (see `.env.example` for the required variables).
    -   Uses `InMemorySessionService` and `InMemoryArtifactService` for session and artifact management.

## Requirements

-   Python 3.9+
-   Packages listed in `requirements.txt` (install with `pip install -r requirements.txt`)
-   A Google Cloud project with the Gemini API enabled
-   A SQLite database (Northwind database is used in the example)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS and Linux
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the environment variables:**

    -   Create a `.env` file in the `sql_assistant_agent` directory.
    -   Add the following variables to the `.env` file:

        ```dotenv
        GOOGLE_GENAI_USE_VERTEXAI=FALSE
        GOOGLE_API_KEY=<Your_Google_API_Key>
        DB_PATH=<Path_to_your_SQLite_database>
        ```

        **Note:** Replace `<Your_Google_API_Key>` with your actual Google API key and `<Path_to_your_SQLite_database>` with the path to your SQLite database file.  An example `.env` file is provided.

5.  **Run the application:**

    ```bash
    python sql_assistant_agent/agent.py
    ```

Alternatively, you can run the application using the following command:

```bash
adk api_server
```

or 

```bash
adk web --reload_agents
```

