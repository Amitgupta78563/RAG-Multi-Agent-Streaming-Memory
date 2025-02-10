import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent import agent_executor

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_scratchpad" not in st.session_state:
    st.session_state.agent_scratchpad = []

def format_chat_history():
    """Format chat history as a string to provide proper context."""
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])

def stream_response(agent_executor, user_input):
    """Stream token-by-token response while preserving history."""
    
    # Prepare the input with chat history
    formatted_history = format_chat_history()
    full_input = f"Conversation so far:\n{formatted_history}\nUser: {user_input}"

    for step in agent_executor.stream({
        "input": full_input,  # Ensure history is included in the input
        "chat_history": st.session_state.chat_history, 
        "agent_scratchpad": st.session_state.agent_scratchpad
    }):
        if isinstance(step, str):
            for token in step.split():
                yield token + " "
        elif isinstance(step, dict) and "output" in step:
            for token in step["output"].split():
                yield token + " "

        # Append intermediate steps to scratchpad
        st.session_state.agent_scratchpad.append(step)

# Streamlit UI
def main():
    st.title("🚀 AI RAG Multi-Agent Streaming Chatbot with Memory")

    # Display previous chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Store user message in history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input, unsafe_allow_html=True)

        with st.chat_message("assistant"):
            response_area = st.empty()
            response_text = ""

            for token in stream_response(agent_executor, user_input):
                response_text += token
                response_area.markdown(response_text, unsafe_allow_html=True)

            # Store assistant response in history
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()

