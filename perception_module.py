from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch

class DuckDetector:
    """
    Uses OpenAI's CLIP model to detect specific objects in an image 
    based on text descriptions.
    """
    def __init__(self):
        print("INFO: Loading CLIP Model (this may take a few seconds)...")
        # Load model and processor once at startup
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Define what we are looking for
        self.labels = ["a yellow duck", "a wall", "an empty floor", "a robot"]
        print("INFO: CLIP Model loaded successfully.")

    def detect(self, rgb_array):
        """
        Analyzes the RGB image and returns the probabilities of the labels.
        """
        # Convert numpy array (from PyBullet) to PIL Image
        image = Image.fromarray(rgb_array)

        # Prepare inputs for CLIP
        inputs = self.processor(
            text=self.labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )

        # Run inference (no_grad to save memory/speed)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Get the highest probability label
        probs_np = probs.numpy()[0]
        best_idx = probs_np.argmax()
        predicted_label = self.labels[best_idx]
        confidence = probs_np[best_idx]

        return predicted_label, confidence, dict(zip(self.labels, probs_np))