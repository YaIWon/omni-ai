"""
COMPLETE LANGUAGE MODEL IMPLEMENTATION WITH FINE-TUNING
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import accelerate

@dataclass
class ModelConfig:
    """Complex model configuration"""
    model_name: str = "microsoft/phi-2"
    quantization: str = "4bit"  # "none", "4bit", "8bit"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    use_flash_attention: bool = True

class AdvancedLanguageModel:
    """Complete language model with training capability"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None
        self.device = self._detect_device()
        self.conversation_memory = ConversationMemory()
        self.knowledge_graph = KnowledgeGraph()
        
        # Load or initialize
        self._initialize_model()
    
    def _detect_device(self) -> str:
        """Detect and configure device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _initialize_model(self):
        """Initialize model with advanced configuration"""
        print(f"[MODEL] Initializing {self.config.model_name} on {self.device}")
        
        # Configure quantization
        bnb_config = None
        if self.config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA for efficient fine-tuning
        self._configure_lora()
        
        print(f"[MODEL] Model loaded: {self.model.num_parameters():,} parameters")
        print(f"[MODEL] Device: {self.device}, Quantization: {self.config.quantization}")
    
    def _configure_lora(self):
        """Configure LoRA for parameter-efficient fine-tuning"""
        if self.config.target_modules is None:
            # Auto-detect target modules
            self.config.target_modules = self._find_target_modules()
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 500,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 top_k: int = 50,
                 repetition_penalty: float = 1.1,
                 do_sample: bool = True,
                 num_beams: int = 1,
                 num_return_sequences: int = 1) -> Dict:
        """Advanced text generation with multiple strategies"""
        
        # Encode input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        # Generation configuration
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        # Extract new text
        results = []
        for text in generated_texts:
            # Remove prompt from response
            if prompt in text:
                response = text[len(prompt):].strip()
            else:
                response = text.strip()
            
            results.append(response)
        
        # Analyze generation
        analysis = self._analyze_generation(prompt, results[0])
        
        return {
            "responses": results,
            "analysis": analysis,
            "generation_config": generation_config,
            "tokens_generated": sum(len(r.split()) for r in results)
        }
    
    def train(self, 
              training_data: List[Dict],
              validation_data: List[Dict] = None,
              epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100) -> Dict:
        """Fine-tune model on custom data"""
        
        # Prepare datasets
        train_dataset = FineTuningDataset(training_data, self.tokenizer)
        val_dataset = FineTuningDataset(validation_data, self.tokenizer) if validation_data else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./models/fine_tuned",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.device == "cuda",
            push_to_hub=False,
            report_to="none"
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Save
        self.trainer.save_model()
        self.tokenizer.save_pretrained("./models/fine_tuned")
        
        return {
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
            "model_saved": True,
            "save_path": "./models/fine_tuned"
        }
    
    def continual_learning(self, new_data: List[Dict]):
        """Continual learning without catastrophic forgetting"""
        # Implement elastic weight consolidation or replay buffers
        pass
    
    def _analyze_generation(self, prompt: str, response: str) -> Dict:
        """Analyze generated text"""
        from collections import Counter
        import re
        
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences if s.strip()]),
            "vocabulary_richness": len(set(words)) / max(len(words), 1),
            "coherence_score": self._calculate_coherence(response),
            "relevance_score": self._calculate_relevance(prompt, response)
        }
    
    def save_state(self, path: str):
        """Save complete model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'conversation_memory': self.conversation_memory,
            'knowledge_graph': self.knowledge_graph
        }, path)
    
    def load_state(self, path: str):
        """Load saved state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.conversation_memory = checkpoint.get('conversation_memory', ConversationMemory())
        self.knowledge_graph = checkpoint.get('knowledge_graph', KnowledgeGraph())

class FineTuningDataset(Dataset):
    """Dataset for fine-tuning"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

class ConversationMemory:
    """Memory for conversation context"""
    
    def __init__(self, max_length: int = 10):
        self.memory = []
        self.max_length = max_length
        self.embeddings = []
    
    def add(self, role: str, content: str, embedding: np.ndarray = None):
        """Add conversation turn"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding
        }
        self.memory.append(entry)
        
        if len(self.memory) > self.max_length:
            self.memory.pop(0)
    
    def get_context(self, window: int = 5) -> str:
        """Get conversation context"""
        recent = self.memory[-window:] if window > 0 else self.memory
        return "\n".join([f"{e['role']}: {e['content']}" for e in recent])

class KnowledgeGraph:
    """Knowledge graph for storing learned information"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.node_counter = 0
    
    def add_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0):
        """Add fact to knowledge graph"""
        sub_id = self._get_node_id(subject)
        obj_id = self._get_node_id(object_)
        
        edge_id = f"{sub_id}-{predicate}-{obj_id}"
        self.edges[edge_id] = {
            "subject": sub_id,
            "predicate": predicate,
            "object": obj_id,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def query(self, subject: str = None, predicate: str = None, object_: str = None):
        """Query knowledge graph"""
        results = []
        for edge_id, edge in self.edges.items():
            if subject and self.nodes[edge["subject"]]["name"] != subject:
                continue
            if predicate and edge["predicate"] != predicate:
                continue
            if object_ and self.nodes[edge["object"]]["name"] != object_:
                continue
            results.append(edge)
        return results
    
    def _get_node_id(self, name: str) -> str:
        """Get or create node ID"""
        for node_id, node in self.nodes.items():
            if node["name"] == name:
                return node_id
        
        node_id = f"node_{self.node_counter}"
        self.nodes[node_id] = {
            "id": node_id,
            "name": name,
            "created": datetime.now().isoformat()
        }
        self.node_counter += 1
        return node_id
