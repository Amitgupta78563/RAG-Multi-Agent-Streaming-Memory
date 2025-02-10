 
# 🚀 AI RAG Multi-Agent Streaming Chatbot with Memory

This project implements an **AI-powered chatbot** with **Retrieval-Augmented Generation (RAG)**, streaming responses, and a **memory system** for persistent chat history. It integrates **LangChain agents**, **LLM tools**, and **Streamlit UI** for an interactive chat experience.

## 📌 Features

✅ **Retrieval-Augmented Generation (RAG)** using Chroma Vector Store  
✅ **Multi-Agent System** with LangChain  
✅ **Persistent Chat Memory** for context retention  
✅ **Streaming Responses** for a real-time chat experience  
✅ **Multiple AI Tools** for chatting, summarization, and retrieval  
✅ **Modular Code Structure** for scalability and maintainability  

---

## 📂 Project Structure

```
├── agents
│   ├── agent.py              # Defines LangChain agent with tools
│
├── memory
│   ├── memory.py             # Chat history management class
│
├── models
│   ├── llm_model.py          # Load LLM model
│
├── prompts
│   ├── prompt.py             # Custom prompt design
│
├── tools
│   ├── chat_tool.py          # GPT-based chatbot tool
│   ├── summarization.py      # Summarization tool
│   ├── retrieval.py          # Chroma vector store retrieval
│
├── app
│   ├── main.py               # Streamlit UI for chatbot
│
├── README.md                 # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 🛠️ Installation & Setup

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2️⃣ **Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 🏗️ Implementation Details

### 🔹 **Memory Management** (`memory.py`)
Manages **chat history** persistently, replacing `st.session_state.chat_history`.
```python
class Memory:
    def __init__(self):
        self.history = []
    def add_user_input(self, input_text):
        self.history.append({"role": "user", "text": input_text})
    def add_assistant_response(self, response_text):
        self.history.append({"role": "assistant", "text": response_text})
    def get_history(self):
        return self.history[-10:]
```

### 🔹 **GPT-Based Chatbot Tool** (`chat_tool.py`)
Uses **DialoGPT** to generate responses while incorporating chat history.
```python
class GPTTool:
    def __init__(self, memory):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.memory = memory
    def execute(self, input_text):
        self.memory.add_user_input(input_text)
        history_text = self.memory.format_history()
        inputs = self.tokenizer.encode(history_text + " " + input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=1000)
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.memory.add_assistant_response(response_text)
        return response_text
```

### 🔹 **LangChain Agent with Memory** (`agent.py`)
Integrates **memory** into the agent for persistent conversations.
```python
from memory import Memory
memory = Memory()
chat_tool = GPTTool(memory)
agent = create_react_agent(load_llm(), [Tool(name="Chat", func=chat_tool.execute)])
```

### 🔹 **Streamlit UI with Streaming Responses** (`main.py`)
Integrates **memory-based chat history** for an interactive user experience.
```python
import streamlit as st
from agents.agent import agent_executor
from memory import Memory
memory = Memory()

def stream_response(agent_executor, user_input):
    for step in agent_executor.stream({"input": user_input, "chat_history": memory.get_history()}):
        if isinstance(step, str):
            yield step + " "

def main():
    st.title("🚀 AI RAG Multi-Agent Streaming Chatbot with Memory")
    user_input = st.chat_input("Type your message...")
    if user_input:
        memory.add_user_input(user_input)
        response_text = "".join(stream_response(agent_executor, user_input))
        memory.add_assistant_response(response_text)
```

---

## 🚀 Running the Chatbot
```sh
streamlit run app/main.py
```

---

## 📌 Future Enhancements
✅ Store chat history in a database for **persistent storage**  
✅ Add support for **multiple LLM models**  
✅ Fine-tune **response generation** for better accuracy  

---


## 🎯 Contributing
We welcome contributions! Feel free to submit a PR or raise an issue. 😊

