import ast
from contextlib import aclosing
import json
import asyncio
from typing import Optional, List, Dict, TYPE_CHECKING
from core.agent import Agent
from core.my_llm import HelloAgentsLLM
from core.config import Config
from core.message import Message
from .simple_agent import SimpleAgent

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

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
            **kwargs: LLM parameters

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
    def __init__(
            self,
            llm_client: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            tool_registry: Optional["ToolRegistry"] = None,
            enable_tool_calling: bool = True,
            max_tool_iteration: int = 3,
    ):
        self.llm_client = llm_client
        self.system_prompt = (
            system_prompt or "You are a professional AI executor, your task is solving problem step by step strictly according to the given plan, focus on current steps only, and give final answer for the step"
            ""
        )
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling
        self.max_tool_iteration = max_tool_iteration

    def _format_plan(self,plan: List[str]) -> str:
        """Initial plan list"""
        return "\n".join([f"{i}. {step}" for i, step in enumerate(plan, 1)])

    def _format_history(self,history: List[Dict[str,str]]) -> str:
        """Initial history log"""
        return "\n\n".join(
            [
                f"Step {i}: {h['step']}\n Result: {h['result']}"
                for i, h in enumerate(history,1)
            ]
        )

    def step_prompt(slef,question,plan,history,step):
        return (
            f"Original Task: {question}\n Full Plan: \n {self._format_plan(plan)}"
            f"Completed Step: \n{self._format_history(history) or "none"}\n"
            f"Current Step: {step}\n Please execture current step and provide result"
        )

    def execute(self,question,plan,**kwargs):
        if not plan:
            raise ValueError("Execution plan can't be null")
        hisotry = []
        for step in plan:
            result = self._execute_step(
                self.step_prompt(question,plan,hisotry,step),**kwargs
            )
            hisotry.append({"step":step, "result":result})
        return hisotry[-1]["result"]

    def _execute_step(self,context,**kwargs):
        #Indipendent executor registor, with no tool registor
        config = Config(
            trace_enabled=False,
            session_enabled=False,
            skills_enabled=False,
            subagent_enabled=False,
            todowrite_enabled=False,
            devlog_enabled=False,
        )

        agent = SimpleAgent(
            "executor",
            self.llm_client,
            self.system_prompt,
            config,
            tool_registry = self.tool_registry,
            enable_tool_calling = self.enable_tool_calling
            max_tool_iterations = self.max_tool_iterations
        )

        answer = agent.run(context,**kwargs)
        if agent.last_run_status != "completed":
            raise RuntimeError(f"Step not finish {agent.last_run.status}")
        
        

    

        