from transformers import pipeline

class SummarizationTool:
    """Summarization tool using transformers pipeline."""
    def __init__(self):
        self.summarizer = pipeline("summarization")

    def execute(self, input_text):
        """Generate summary of the input text."""
        summary = self.summarizer(input_text, max_length=1000, min_length=30, do_sample=False)
        return summary[0]['summary_text']

