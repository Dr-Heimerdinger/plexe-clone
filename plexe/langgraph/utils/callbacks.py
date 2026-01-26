from typing import Optional, Dict, Any, List
from plexe.langgraph.utils.emitters import BaseEmitter, ConsoleEmitter
from plexe.langgraph.utils.progress import AgentProgress

class ChainOfThoughtCallback:
    """Callback for capturing chain-of-thought from LangGraph agents."""
    
    def __init__(self, emitter: Optional[BaseEmitter] = None):
        self.emitter = emitter or ConsoleEmitter()
        self.progress = AgentProgress()
        self.thoughts: List[Dict[str, str]] = []
    
    def on_agent_start(self, agent_name: str, **kwargs):
        """Called when an agent starts processing."""
        self.progress.current_agent = agent_name
        self.emitter.emit_agent_start(agent_name)
    
    def on_agent_action(self, agent_name: str, action: str, **kwargs):
        """Called when an agent takes an action."""
        self.thoughts.append({"agent": agent_name, "action": action})
        self.emitter.emit_thought(agent_name, action)
    
    def on_tool_start(self, agent_name: str, tool_name: str, args: Dict[str, Any], **kwargs):
        """Called when a tool is invoked."""
        self.emitter.emit_tool_call(agent_name, tool_name, args)
    
    def on_tool_end(self, agent_name: str, tool_name: str, result: Any, **kwargs):
        """Called when a tool completes."""
        pass
    
    def on_agent_end(self, agent_name: str, result: str, **kwargs):
        """Called when an agent completes processing."""
        self.emitter.emit_agent_end(agent_name, result)
    
    def on_llm_start(self, agent_name: str, prompt: str, **kwargs):
        """Called when LLM inference starts."""
        self.emitter.emit_thought(agent_name, "Processing...")
    
    def on_llm_end(self, agent_name: str, response: str, **kwargs):
        """Called when LLM inference completes."""
        pass


def create_langchain_callbacks(emitter: BaseEmitter, agent_name: str):
    """Create LangChain-compatible callbacks from an emitter."""
    from langchain_core.callbacks import BaseCallbackHandler
    
    class LangChainEmitterCallback(BaseCallbackHandler):
        def __init__(self, emitter: BaseEmitter, agent_name: str):
            self.emitter = emitter
            self.agent_name = agent_name
        
        def on_llm_start(self, serialized, prompts, **kwargs):
            self.emitter.emit_thought(self.agent_name, "Analyzing request...")
        
        def on_llm_end(self, response, **kwargs):
            pass
        
        def on_tool_start(self, serialized, input_str, **kwargs):
            tool_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "tool"
            self.emitter.emit_tool_call(self.agent_name, tool_name, {})
        
        def on_tool_end(self, output, **kwargs):
            pass
        
        def on_chain_start(self, serialized, inputs, **kwargs):
            pass
        
        def on_chain_end(self, outputs, **kwargs):
            pass
    
    return [LangChainEmitterCallback(emitter, agent_name)]

