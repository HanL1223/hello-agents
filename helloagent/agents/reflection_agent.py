from typing import Optional, List, Dict, Any
from core.agent import Agent
from core.hello_agents import HelloAgentsLLM
from core.config import Config
from core.message import Message


DEFAULT_PROMPTS = {
    "initial": """
Please complete the task according to the following requirements:

Task: {task}

Please provide a complete and accurate response.
""",
    "reflect": """
Please carefully review the following response and identify any potential issues or areas for improvement:

# Original Task:
{task}

# Current Response:
{content}

Please analyze the quality of this response, point out its shortcomings, and provide specific suggestions for improvement.
If the response is already good, reply with "No improvement needed".
""",
    "refine": """
Please improve your response based on the feedback provided:

# Original Task:
{task}

# Previous Response:
{last_attempt}

# Feedback:
{feedback}

Please provide an improved response.
"""
}