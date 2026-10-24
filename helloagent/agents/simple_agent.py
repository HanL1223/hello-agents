from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message,ToolRegistry
import re

class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True
    ):
        super.__init__(name,llm,system_prompt,config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"Init complenete, tool calling {enable_tool_calling}")


    def run(self,input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        running method for simple converstation with tool calling optional
        """

        print(f"{self.name} processing {input_text}")

        #message list
        messages = []

        # System message with tool info

        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role":"system", "content":enhanced_system_prompt})

        # Adding historial message
        for msg in self._history:
            messages.append({"role":msg.role, "content":msg.content})

        messages.append({"role":"user", "content":input_text})

        #Simple conversation without tool call
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text,"user"))
            self.add_message(Message(response, "assistant"))
            print(f"Response completed")
            return response

        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def _run_with_tools(self,messages: list, input_text: str, max_tool_iterations:int , **kwargs) -> str:
        """Running logic supporting tool call"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # Call LLM
            response = self.llm.invoke(messages, **kwargs)

            # Check if tool call allowed
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"{len(tool_calls)} of tool call detected")
                
