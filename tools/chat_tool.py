from transformers import AutoTokenizer, AutoModelForCausalLM

class GPTTool:
    """Chatbot tool using DialoGPT."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    def execute(self, input_text, chat_history=[]):
        """Generate conversational response."""
        history_text = " ".join([msg["text"] for msg in chat_history[-5:]])
        inputs = self.tokenizer.encode(history_text + " " + input_text + self.tokenizer.eos_token, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

