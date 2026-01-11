import torch
from torchvision import transforms
from PIL import Image
import config
from training.model_factory import create_model
from device_helper import get_device

class Predictor:
    """Unified prediction interface for final models"""
    
    def __init__(self, model_name: str = "resnet50"):
        self.model_name = model_name
        self.device, _ = get_device()
        self.model, self.classes = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
        ])
        
    def _load_model(self):
        """Load final deployment model"""
        model_path = config.FINAL_MODELS_DIR / f"{self.model_name}_final.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Final model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location="cpu")
        classes = checkpoint["classes"]
        
        model = create_model(self.model_name, len(classes))
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        
        print(f"[INFO] Loaded {self.model_name} from {model_path}")
        return model, classes
    
    def predict(self, image: Image.Image):
        """Predict class from PIL image"""
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            pred_idx = out.argmax(dim=1).item()
            confidence = probs[pred_idx].item()
        
        # Top 3 predictions
        top_probs, top_indices = torch.topk(probs, min(3, len(self.classes)))
        top_predictions = [
            {"class": self.classes[idx], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            "prediction": self.classes[pred_idx],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model": self.model_name
        }