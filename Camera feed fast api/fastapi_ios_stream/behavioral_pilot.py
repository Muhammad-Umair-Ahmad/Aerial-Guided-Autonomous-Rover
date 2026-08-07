import os
import torch
from torchvision import transforms
from PIL import Image
import io
import base64

class BehavioralPilot:
    def __init__(self, model_path="models/behavioral_clone.pth"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.transform = None
        self.idx_to_cmd = {
            0: "forward",
            1: "reverse",
            2: "left",
            3: "right",
            4: "stop"
        }

    def load_model(self):
        print("[AI PILOT] Loading Behavioral Cloning model...")
        if not os.path.exists(self.model_path):
            print(f"[AI PILOT] Error: Model not found at {self.model_path}")
            return False

        try:
            from train_model import RoverCNN
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = RoverCNN(num_classes=5).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("[AI PILOT] Model loaded successfully.")
            return True
        except Exception as e:
            print(f"[AI PILOT] Failed to load model: {e}")
            return False

    def predict_command(self, image_b64: str) -> str:
        if self.model is None:
            return "stop"
            
        try:
            if "," in image_b64:
                _, encoded = image_b64.split(",", 1)
            else:
                encoded = image_b64
                
            img_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs.data, 1)
                cmd_idx = predicted.item()
                
            return self.idx_to_cmd.get(cmd_idx, "stop")
            
        except Exception as e:
            print(f"[AI PILOT] Inference error: {e}")
            return "stop"
