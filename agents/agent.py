from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor, Tool
from models.llm_model import load_llm
from prompts.prompt import prompt
from tools.chat_tool import GPTTool
from tools.summarization import SummarizationTool
from tools.retrieval import rag_chain


chat_tool = GPTTool()
summarization_tool = SummarizationTool()

# 7. Define the tools in the agent
tools = [
    Tool(
        name="Chroma Vector Store",
        func=lambda input, **kwargs:rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", []),
            "agent_scratchpad": kwargs.get("agent_scratchpad", [])}),
        # func=lambda input, **kwargs: qa_chain.invoke({"input": input, "context": retriever}),
        description="Use it to lookup information from the Chroma vector store"
    ),
    Tool(
        name="GeneralChatTool",
        func=chat_tool.execute,
        description="A chatbot tool that generates conversational responses using DialoGPT."
    ),
    Tool(
        name="SummarizationTool",
        func=summarization_tool.execute,
        description="A summarization tool that shortens long text."
    ),
]

# Define Agent
agent = create_react_agent(load_llm(), tools, prompt)
agent_executor = AgentExecutor(tools=tools,
                                agent=agent,
                                handle_parsing_errors=True,
                                verbose=True
                               )