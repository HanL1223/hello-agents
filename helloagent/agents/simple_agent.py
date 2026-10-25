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
        super().__init__(name,llm,system_prompt,config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"Init complenete, tool calling {self.enable_tool_calling}")


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

                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    #
                    clean_response = clean_response.replace(call["original"],"")

                # append message with tool result
                messages.append({"role": "assistant", "content": clean_response})
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"Tool run result:\n{tool_results_text}\n\n provide full answer base on it"})

                current_iteration += 1
                continue

            final_response = response
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"{self.name} response completed")
        return final_response

    def _parse_tool_calls(self,text: str) -> list:
        """interpret tool call in text"""

        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        tool_calls = []

        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name:str, parameters: str) -> str:
        """"""
        if not self.tool_registry:
            return "Error: No tool registry found"

        try:
            if tool_name == "calculator":
                result = self.tool_registry.execute_tool(tool_name,parameters)
            else:
                param_dict = self._parse_tool_parameters(tool_name,parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"No tool found under {tool_name}"
                return tool.run(param_dict)
            return f"Tool call {tool_name} Result: \n {result}"
        except Exception as e:
            return f"Tool call failed {str(e)}"
        
    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数:action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数:key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """添加工具到Agent（便利方法）"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False
    
    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []