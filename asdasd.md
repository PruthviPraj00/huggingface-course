from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(
tools=[DuckDuckGoSearchTool()],
model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
)

web_agent = ToolCallingAgent(
tools=[DuckDuckGoSearchTool(), visit_webpage],
model=model,
max_steps=10,
name="search",
description="Runs web searches for you.",
)

manager_agent = CodeAgent(
tools=[],
model=model,
managed_agents=[web_agent],
additional_authorized_imports=["time", "numpy", "pandas"],
)

from smolagents import CodeAgent, E2BSandbox

agent = CodeAgent(
tools=[], model=model, sandbox=E2BSandbox(), additional_authorized_imports=["numpy"]
)

from smolagents import ToolCallingAgent

agent = ToolCallingAgent(
tools=[custom_tool],
model=model,
max_steps=5,
name="tool_agent",
description="Executes specific tools based on input",
)

from smolagents import HfApiModel, LiteLLMModel

# Hugging Face model

hf_model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Alternative model via LiteLLM

other_model = LiteLLMModel("anthropic/claude-3-sonnet")
