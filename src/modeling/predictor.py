"""
Model Predictor
Makes predictions using trained models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Makes predictions using trained models."""
    
    def __init__(self, model_path: str):
        """Load a trained model."""
        self.model = None
        self.config = None
        self.load_model(model_path)
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Try to load config
        config_path = path.with_name(f"{path.stem}_config.pkl")
        if config_path.exists():
            self.config = joblib.load(config_path)
    
    def predict(self, X: pd.DataFrame, 
               return_probabilities: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X: Features
            return_probabilities: Whether to return class probabilities
        
        Returns:
            predictions, probabilities (if requested)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        predictions = self.model.predict(X)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        
        return predictions, None
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                               confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Make predictions with confidence filtering.
        
        Args:
            X: Features
            confidence_threshold: Minimum probability to make a prediction
        
        Returns:
            DataFrame with predictions and confidences
        """
        predictions, probabilities = self.predict(X, return_probabilities=True)
        
        # Get max probability for each prediction
        max_proba = probabilities.max(axis=1)
        
        # Filter by confidence
        confident_mask = max_proba >= confidence_threshold
        
        # Create result DataFrame
        result = pd.DataFrame({
            'prediction': predictions,
            'confidence': max_proba,
            'is_confident': confident_mask,
            'prob_sell': probabilities[:, 0],
            'prob_hold': probabilities[:, 1],
            'prob_buy': probabilities[:, 2],
        }, index=X.index)
        
        # Set low-confidence predictions to HOLD (1)
        result.loc[~confident_mask, 'prediction'] = 1
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage would require a trained model
    print("ModelPredictor ready for use")
