"""
ACTUAL LANGUAGE MODEL LOADER
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from pathlib import Path
from typing import Optional

class RawLanguageModel:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.device = self._get_device()
        self.conversation_history = []
        
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the actual language model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device in ["cuda", "mps"] else None,
                trust_remote_code=True
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            # Fallback to local download
            return self._download_and_load_local()
    
    def process(self, text: str, max_length: int = 500) -> str:
        """Process text through language model"""
        if self.generator is None:
            return f"[Model not loaded] {text}"
        
        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {text}")
            
            # Prepare prompt with history
            prompt = "\n".join(self.conversation_history[-5:]) + "\nAI:"
            
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )[0]['generated_text']
            
            # Extract AI response
            if "AI:" in response:
                ai_response = response.split("AI:")[-1].strip()
            else:
                ai_response = response.replace(prompt, "").strip()
            
            # Add to history
            self.conversation_history.append(f"AI: {ai_response}")
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return ai_response
            
        except Exception as e:
            return f"[Model error] {str(e)}"
    
    def fine_tune(self, training_data: list):
        """Fine-tune model on new data"""
        # Implementation would go here
        pass
