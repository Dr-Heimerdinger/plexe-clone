"""
Placeholder predictor template for backward compatibility.
The LangGraph system generates code directly without templates.
"""

class Predictor:
    """Base predictor interface."""
    
    def __init__(self):
        pass
    
    def predict(self, input_data):
        raise NotImplementedError("Subclasses must implement predict()")
    
    def load(self, path):
        raise NotImplementedError("Subclasses must implement load()")
    
    def save(self, path):
        raise NotImplementedError("Subclasses must implement save()")
