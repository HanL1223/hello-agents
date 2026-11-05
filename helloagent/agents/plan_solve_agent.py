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
    """Executor for running according to plan"""

    def __init__(self, llm_client:HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question:str, plan = list[str], **kwargs) -> str:
        """
        Execute as per given plan

        Args:
            questions: Original question
            plan: Execute plan genereated from planner
            **kwargs: LLM Parameters
        Returns:
            Final answer 
        """
        history = ""
        final_answer = ""

        print("\n--Executing Plan--")
        for i,step in enumerate(plan,1):
            print(f"Executing step {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question = question,
                plan = plan,
                history = history if history else "None",
                current_step = step
            )
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.invoke(messages,**kwargs) or ""

            history += f"Step {i}: {step}\n Result: {response_text}\n\n"
            final_answer = response_text
            print(f"Step {i} completed, result: {final_answer}")

        return final_answer
    
class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent breaks down question into actionable steps, in particular

    - Break complex question into simple,actionable steps
    - Execute as per plan
    - Maintain execute history and context
    - Provide final answer

    Suitable for multi-step inference, math and complex analysis
    """
    def __init__(
            self,
            name:str,
            llm:HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            custom_prompts:Optional[Dict[str,str]] = None
    ):
        """
        Init agent

        Args:
            name:agent name
            llm: llm client
            system_prompt: system prompt
            config: configruation object
            custome_prompts = customer prompt template {"planner": "", "executor": ""}
        """
        super().__init__(name,llm,system_prompt,config)

        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")

        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, input_text:str, **kwargs) -> str:
        """
        Run plan and solve agent

        args:
            input_text: problem needs fix
            **kwargs: Other parameters 
        Return:
            Final answer
        """

        print(f"\n{self.name} Begin processing question {input_text}")

        #1.generate plan
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "No valid action plan, abord"
            print(f"\n--Task End--\n{final_answer}")

            self.add_message(Message(input_text,"user"))
            self.add_message(Message(final_answer,"assistant"))
            return final_answer

        #2. execute plan
        final_answer = self.executor.execute(input_text,plan,**kwargs)
        print(f"Task completed\n final answer: {final_answer}")

        # Append to history
        self.add_message(Message(input_text,"user"))
        self.add_message(Message(final_answer,"assistant"))

        return final_answer
