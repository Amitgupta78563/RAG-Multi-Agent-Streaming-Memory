class Memory:
    """Manage chat history."""
    def __init__(self):
        self.history = []

    def add_user_input(self, input_text):
        self.history.append({"role": "user", "text": input_text})

    def add_assistant_response(self, response_text):
        self.history.append({"role": "assistant", "text": response_text})

    def get_history(self):
        return self.history

