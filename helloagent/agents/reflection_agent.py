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



class Memory:
    """
    Simple short term memory for store agent action and reflection trail
    """

    def __init__(self):
        self.records: List[Dict[str,Any]] = []

    def add_record(self,record_type: str, content: str):
        """Adding an record to memory"""
        self.records.append({"type":record_type, "content":content})
        print(f"Memory updated with type {record_type}")

    def get_trajectory(self)-> str:
        """Format all memeory into 1 string"""
        trajectory = ""
        for record in self.records:
            if record["type"] =='execution':
                trajectory += f"--Previous attemp(Code)--\n{record['content']}\n\n"
            elif record["type"] =="reflection":
                trajectory += f"--Reviewer feedback--\n{record['content']}\n\n"
            return trajectory.strip()


    def get_last_execution(self) -> str:
        """Retrieve recent execution result"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ''

class ReflectionAgent(Agent):
    """
    Reflection Agent - An intelligent agent for self-reflection and iterative optimization

    This Agent can:
    1. Execute the initial task.
    2. Self-reflect on the results.
    3. Improve the results based on its reflections.
    4. Iteratively refine the results until they meet expectations.

    It is particularly suitable for tasks that require iterative refinement,
    such as code generation, document writing, and analytical reporting.

    Supports prompt templates for 
    various professional domains.
    Users can customize their own templates 
    or use the built-in templates.
    """
    def __init__(
      self,
      name:str,
      llm:HelloAgentsLLM,
      system_prompt:Optional[str] = None,
      config: Optional[Config] = None,
      max_iterations:int  = 3,
      custom_prompts: Optional[dict[str,str]] = None      
    ):
        """
        Initialize the ReflectionAgent.

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt
            config: Configuration object
            max_iterations: Maximum number of iterations
            custom_prompts: Custom prompt templates {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name,llm,system_prompt,config)
        self.max_iterations = max_iterations
        self.memory = Memory()
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS


    def run(self,input_text:str, **kwargs) -> str:
        """
        
        """
        print(f"\n{self.name} start processing task {input_text}")

        #Reset memory
        self.memory = Memory()

        #1.Init
        print("\n--start first try--")
        initial_prompt = self.prompts['initial'].format(task = input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution",initial_result)

        #2. Iteration loop
        for i in range(self.max_iterations):
            print(f"Iteration round {i+1}/{self.max_iterations}")

            # a. reflection
            print("--Begin reflection--")
            last_result = self.memory.get_last_execution()
            reflection_prompt = self.prompts["reflection"].format(
                task = input_text,
                content = last_result
            )
            feedback = self._get_llm_response(reflection_prompt,**kwargs)
            self.memory.add_record("reflection",feedback)
            #b.check if need to stop
            if "No improvement needed" in feedback.lower():
                print("\n Reflection finish with no imporvement require")
                break

            #c. optimisation
            print("\n Optimising")
            refine_prompt = self.prompts["refine"].format(
                task = input_text,
                last_attemp = last_result,
                feedback = feedback
            )
            refined_result = self._get_llm_response(reflection_prompt,**kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--Missing Completed--\nFinal Result:\n{final_result}")


        #Add to history
        self.add_message(Message(input_text,"user"))
        self.add_message(Message(final_result,"assistant"))

        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """call LLM and get response"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
