import csv
from datetime import datetime
from pathlib import Path
import config

class InferenceLogger:
    """Log all predictions made during live inference"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = config.INFERENCE_LOGS_DIR / f"session_{self.session_id}.csv"
        self._init_log_file()
        
    def _init_log_file(self):
        """Create log file with headers"""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model", "prediction", "confidence", 
                           "top3_classes", "top3_confidences", "source"])
            
    def log_prediction(self, model: str, prediction: str, confidence: float, 
                      top_predictions: list, source: str = "webcam"):
        """Log a single prediction"""
        timestamp = datetime.now().isoformat()
        top3_classes = [p["class"] for p in top_predictions[:3]]
        top3_confs = [p["confidence"] for p in top_predictions[:3]]
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, model, prediction, confidence,
                "|".join(top3_classes), "|".join(map(str, top3_confs)), source
            ])
    
    def get_log_path(self) -> Path:
        return self.log_file