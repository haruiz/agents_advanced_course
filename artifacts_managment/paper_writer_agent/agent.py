import os

import pypandoc
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import BaseModel

DEFAULT_MODEL=os.getenv("DEFAULT_MODEL", "gemini-flash-latest")

async def create_pdf(filename: str, tool_context: "ToolContext", **kwargs) -> dict:
    """
    Create a PDF from paper_content sections using Pandoc (via pypandoc).
    Assumes text is already correctly encoded.
    """
    try:
        paper_content = kwargs.get("paper_content") or tool_context.state.get("paper_content", {})

        # Accept dict or pydantic models; fall back to string
        if hasattr(paper_content, "model_dump"):
            sections = paper_content.model_dump()
        elif hasattr(paper_content, "dict"):
            sections = paper_content.dict()
        elif isinstance(paper_content, dict):
            sections = paper_content
        else:
            sections = {"Content": str(paper_content)}

        # Build Markdown
        md_parts = []
        for title, body in sections.items():
            if body is None:
                continue
            md_parts.append(f"## {title}\n\n{body}\n")
        markdown = "\n".join(md_parts).strip() or " "

        # Convert Markdown -> PDF (writes directly to filename)
        pypandoc.convert_text(
            markdown,
            to="pdf",
            format="md",
            outputfile=filename,
            extra_args=["--standalone"]
        )

        # Read bytes back and save as artifact
        with open(filename, "rb") as f:
            pdf_bytes = f.read()

        await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        )

        return {"status": "success", "message": f"PDF '{filename}' created successfully."}

    except Exception as e:
        return {"status": "error", "message": f"Failed to create PDF: {e}"}

search_agent = Agent(
    model=DEFAULT_MODEL,
    name='search_agent',
    description='A helpful assistant for searching the web.',
    instruction=(
        'Use the Google Search tool if available to find relevant information on the web, '
        "otherwise use your internal knowledge to find relevant information. "
        "if any feedback is provided, refine the search query accordingly and"
        "return the search results and references."
    ),
    #tools=[google_search],
    output_key='search_results',
)

class PaperDocumentStructure(BaseModel):
    Abstract: str
    Introduction: str
    Methodology: str
    Results: str
    Discussion: str
    Conclusion: str

student_agent = Agent(
    model=DEFAULT_MODEL,
    name='student_agent',
    description='A helpful assistant for researching topics.',
    instruction=(
        "You are a research assistant agent. Your task is to generate a high-quality academic paper based on the provided search results. "
        "Extract key information and insights relevant to the research topic. "
        "If additional information is required, provide feedback to the search agent to refine the query. "
        "The final paper should be at least 1000 words and written at a level suitable for submission to a top-tier conference. "
        "Structure the output as a JSON object, where each key corresponds to a standard section of a research paper: "
        "`Abstract`, `Introduction`, `Methodology`, `Results`, `Discussion`, and `Conclusion`. "
        "Each section should contain well-developed, original text appropriate to its purpose."
    ),
    output_schema=PaperDocumentStructure,
    output_key='paper_content',
)

postdoc_agent = LoopAgent(
    name='postdoc_agent',
    description='A loop agent that coordinates the search and research agents.',
    max_iterations=1,
    sub_agents=[search_agent, student_agent]
)

postdoc_agent_tool = AgentTool(agent=postdoc_agent)

root_agent = Agent(
    model=DEFAULT_MODEL,
    name='paper_writer_agent',
    description='A helpful assistant for writing research papers.',
    instruction=(
        'You are a paper writing assistant. Your task is to compile the research findings into a well-structured academic paper. '
        'Use the provided research content to create a coherent and comprehensive document. '
        'Finally, save the PDF document using the create_pdf tool'
    ),
    tools=[postdoc_agent_tool, create_pdf],
)
