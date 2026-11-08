import re
from typing import Optional, List, Dict, Any, Tuple
from ..core.agent import Agent
from ..core.hello_agents import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# Default ReAct prompt
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

**Thought:** 分析当前问题，思考需要什么信息或采取什么行动。
**Action:** 选择一个行动，格式必须是以下之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数
5. 输出必须使用以 `Thought:` 和 `Action:` 开头的纯文本行；`Action:` 后可直接写操作，也可在下一行写操作。

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    
    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    
    这是一个经典的Agent范式，特别适合需要外部信息的任务。
    """
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            tool_registry: ToolRegistry,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_steps: int = 5,
            custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        super().__init__(name,llm,system_prompt,config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []

        #Setting Prompt template, user custom first then default
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def run(self,input_text:str, **kwarg) -> str:
        """
        Running React Agent

        Args:
            input_text: User question
            **kwarg: Other parameter
        
        Returns:
            Final Output
        """
        self.current_history = []
        current_steps  = 0

        print(f"\n{self.name} Begin process: {input_text}")

        while current_steps < self.max_steps:
            current_steps += 1
            print (f"Step No:{current_steps}")

            #creating prompt
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools = tools_desc,
                question = input_text,
                history = history_str
            )

            #Calling LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages,**kwarg)

            if not response_text:
                print("Error: LLM did not return")
                break

            thought, action = self._parse_output(response_text)

            if thought:
                print(f"Thinking {thought}")

            if not action:
                print(f"Warning, No valid action detected, end loop")
                break

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"Final Answer{final_answer}")

                self.add_message(Message(input_text,"user"))
                self.add_message(Message(final_answer,"assistant"))

                return final_answer

            tool_name, tool_input = self._parse_action(action)

            if not tool_name or tool_input is None:
                self.current_history.append("Observation: Not valid action type")
                continue
            print(f"Action: {tool_name}[{tool_input}]")

            observation = self.tool_registry.execute_tool(tool_name,tool_input)
            print(f"Observe:{observation}")

            #update history
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        print("Max steps reach")
        final_answer = "Sorry, Mission can't be finish in require steps"

        self.add_message(Message(input_text,"user"))
        self.add_message(Message(final_answer,"assistant"))

        return final_answer

    def _parse_output(self,text: str) -> Tuple[Optional[str],Optional[str]]:
        """Interprete LLM Output,extract thought and action"""
        thought = self._extract_label_value(text, "Thought")
        action = self._extract_label_value(text, "Action")
        
        return thought, action

    @staticmethod
    def _extract_label_value(text: str, label: str) -> Optional[str]:
        """Read a Markdown or plain-text ``Label: value`` response field."""
        match = re.search(
            rf"(?im)^\s*(?:[-+]\s+)?(?:\*\*)?{label}(?:\*\*)?\s*:(?:\*\*)?\s*(.*)$",
            text,
        )
        if not match:
            return None

        value = match.group(1).strip()
        if value:
            return value

        remaining_lines = text[match.end():].splitlines()
        for line in remaining_lines:
            candidate = line.strip().strip("*")
            if candidate:
                return candidate
        return None


    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Interprete Action Text, extract tool name and tool input"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        """Extract action input"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""




