"""
Placeholder feature transformer template for backward compatibility.
The LangGraph system generates code directly without templates.
"""

class FeatureTransformer:
    """Base feature transformer interface."""
    
    def __init__(self):
        pass
    
    def fit(self, data):
        raise NotImplementedError("Subclasses must implement fit()")
    
    def transform(self, data):
        raise NotImplementedError("Subclasses must implement transform()")
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
