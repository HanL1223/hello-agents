import ast
from typing import Optional, List, Dict, Any
from core.agent import Agent
from core.my_llm import HelloAgentsLLM
from core.config import Config
from core.message import Message


# Default planner prompt
DEFAULT_PLANNER_PROMPT = """
You are a top-tier AI planning expert. Your task is to break down complex problems presented by users into an action plan consisting of multiple simple steps.
Ensure that each step in the plan is an independent, executable subtask, and that all steps are arranged in a strict logical order.
Your output must be a Python list, where each element is a string describing a subtask.

Question: {question}

Please strictly follow the format below when outputting your plan:
```python
["Step 1", "Step 2", "Step 3", ...]
```
"""

# Default executor prompt
DEFAULT_EXECUTOR_PROMPT = """
You are a top-tier AI execution expert. Your task is to solve problems step by step, strictly following the given plan.
You will receive the original question, the complete plan, and the steps completed so far along with their results.
Focus exclusively on solving the "current step" and output only the final answer for that step. Do not include any additional explanations or conversation.

# Original Question:
{question}

# Complete Plan:
{plan}

# Historical Steps and Results:
{history}

# Current Step:
{current_step}

Please output only the answer to the "current step":
"""

class Planner:
    """Planner to strip complex questions into simple steps"""
    def __init__(self,llm_client: HelloAgentsLLM, prompt_template: Optional[str]= None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self,question:str, **kwargs) -> List[str]:
        """
        Generate action plan

        Args:
            Question: Question to resolve
            **kwarg: LLM parameters

        Retrun:
            List of steps
        """
        prompt  = self.prompt_template.format(question = question)
        messages = [{"role": "user", "content": prompt}]

        print("Generating Plan")
        response_text = self.llm_client.invoke(messages, **kwargs)
        print(f"Plan Generated: \n{response_text}")

        try:
            # Extract list from python code block
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError,SyntaxError, IndexError) as e:
            print(f" Error when interprete plan: {e}")
            print(f"Original response {response_text}")
            return []
        except Exception as e:
            print(f"unknown error when interprete plan")

class Executor:
    