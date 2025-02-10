from langchain.prompts import PromptTemplate

# 8. Define the agent using the tools
prompt_template = '''
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. Always first in Chroma Vector Store tools. Also, if user query response depends on multiple tools then pass one tools response to another for final response. You should remember the chat history.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, Also before response return which tool is used to generate the response like json.

Begin!

Question: {input}
Thought:{agent_scratchpad}
'''

prompt = PromptTemplate.from_template(prompt_template)
