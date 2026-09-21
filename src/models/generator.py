"""Generative language model wrapper for safety alignment simulation.

Provides BaseLanguageModel, wrapping lightweight autoregressive LLMs (e.g., distilgpt2)
with support for steering prefixes, controlled stochastic decoding, and feature embedding
extraction for RL observation spaces.
"""

from typing import Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseLanguageModel:
    """Lightweight causal language model wrapper for policy environment simulations.

    Designed for fast inference during RL rollout steps. Handles device allocation,
    controlled decoding parameters, and prompt embedding extraction via a dedicated
    sentence encoder (all-MiniLM-L6-v2) for isotropic semantic state representations.

    Attributes:
        model_name: Hugging Face model repository identifier (default: 'distilgpt2').
        device: torch.device indicating execution target ('cuda' or 'cpu').
        tokenizer: Pretrained Hugging Face tokenizer instance.
        model: Pretrained causal LM instance.
        sentence_encoder: Dedicated SentenceTransformer encoder for semantic state representations.
        embedding_dim: Dimensionality of the sentence encoder latent space (384 for all-MiniLM-L6-v2).
    """

    DEFAULT_MODEL: str = "distilgpt2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes the generator with model weights, device configuration, and sentence encoder.

        Args:
            model_name: Hugging Face repo ID. Defaults to 'distilgpt2'.
            device: Explicit device string ('cuda', 'cpu'). If None, auto-detected.
        """
        self.model_name: str = model_name or self.DEFAULT_MODEL

        if device is not None:
            self.device: torch.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and causal language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Ensure pad token exists (GPT-2 family does not define pad_token by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()

        # Dedicated sentence encoder for high-variance, isotropic semantic embeddings
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=str(self.device))
        self.embedding_dim: int = 384

    def generate(
        self,
        prompt: str,
        steering_prompt: str = "",
        max_new_tokens: int = 40,
        min_new_tokens: int = 15,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generates continuation text given a prompt and optional steering prefix.

        Args:
            prompt: Base user instruction or query.
            steering_prompt: Optional prepended guidance/system instruction
                (e.g., safety guardrail or alignment directive).
            max_new_tokens: Maximum number of newly generated tokens.
            min_new_tokens: Minimum number of newly generated tokens to enforce substance.
            temperature: Sampling temperature for softmax over vocabulary logits.
            top_p: Nucleus sampling cumulative probability threshold.
            repetition_penalty: Penalty factor applied to previously generated tokens.
            do_sample: Whether to sample stochastically or decode greedily.

        Returns:
            str: Decoded continuation text excluding the input prompt tokens.
        """
        prompt = prompt.strip()
        steering_prefix = steering_prompt.strip()

        # Structured prompt format with Question/Answer completion cues
        if steering_prefix:
            full_input = f"{steering_prefix}\nQuestion: {prompt}\nAnswer:"
        else:
            full_input = f"Question: {prompt}\nAnswer:"

        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        effective_max_tokens = max(max_new_tokens, min_new_tokens)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=effective_max_tokens,
                temperature=max(0.01, float(temperature)),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off input tokens to retain exclusively the newly generated response
        generated_tokens = output_ids[0, input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response

    def get_embedding(self, text: str) -> np.ndarray:
        """Extracts a normalized semantic sentence embedding using all-MiniLM-L6-v2.

        Used as the continuous state observation vector for the Gymnasium RL environment.

        Args:
            text: Input string to embed.

        Returns:
            np.ndarray: 1D float32 array of shape (embedding_dim,) = (384,).
        """
        embedding = self.sentence_encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

