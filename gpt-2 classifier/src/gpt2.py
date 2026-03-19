"""
GPT-2 Implementation from Scratch

This module implements the GPT-2 transformer architecture using only PyTorch.
No HuggingFace dependencies are allowed in this file.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPT2Config:
    """Configuration class for GPT-2 small model."""
    # Total number of tokens in the vocabulary.
    # Note we use the same vocabulary and tokenizer as OpenAI GPT-2.
    vocab_size: int = 50257
    
    # The maximum context window length is 1024 tokens for GPT-2.
    max_ctx_len: int = 1024
    
    # The model dimension (hidden size) for GPT-2 Small is 768.
    d_model: int = 768
    
    # The dimension of each attention head is d_model / n_head = 768 / 12 = 64.
    d_head: int = 64
    
    # The intermediate dimension of the MLP in GPT-2 Small is 4 times the model dimension.
    # 4 * 768 = 3072
    d_mlp_intermediate: int = 3072
    
    # GPT-2 Small has 12 transformer blocks.
    n_layer: int = 12
    
    # GPT-2 Small has 12 attention heads per transformer block.
    n_head: int = 12
    
    # Total number of label classes for our classification dataset.
    num_labels: int = 20


@dataclass
class CausalLMOutput:
    """Output class for causal language modeling. Contains the logits for all input tokens."""
    logits: Tensor


@dataclass
class ModelOutput:
    """Output class for generation. Contains sequences of input and generated token IDs."""
    sequences: Tensor


@dataclass
class SequenceClassifierOutput:
    """Output class for sequence classification. Contains the logits for each label class."""
    logits: Tensor


class TransposedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = TransposedLinear(768, 2304)
        self.c_proj = TransposedLinear(768, 768)
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, self.config.max_ctx_len, self.config.max_ctx_len, dtype=torch.bool)))

    def forward(self, x, layer_past=None):
        qkv = self.c_attn(x) # → shape [B, T, 2304]
        q, k, v = qkv.split(self.config.d_model, dim=-1) # → each [B, T, 768]
        B, T, _ = x.shape
        q = q.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2) # → [B, n_head, T, d_head] but all changed to 1 not T
        k = k.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2) # → [B, n_head, T, d_head]
        v = v.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2) # → [B, n_head, T, d_head]

        if layer_past is not None:
            pastk, pastv = layer_past
            k = torch.cat([pastk,k], dim=2)
            v = torch.cat([pastv,v], dim=2)

        layer_curr = (k, v)

        T_full = k.shape[2]
        T_new = q.shape[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.config.d_head) # → [B, n_head, T, T]
        attn = attn.masked_fill(self.bias[:, :, T_full-T_new : T_full, :T_full] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v # → [B, n_head, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.config.d_model) # → [B, T_new, 768]
        out = self.c_proj(out)

        return out, layer_curr

class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = TransposedLinear(768, 3072)
        self.c_proj = TransposedLinear(3072, 768)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(768) # pre attention layer norm
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(768) # pre MLP layer norm
        self.mlp = GPT2MLP()

    def forward(self, x, layer_past=None):
        attn_out, curr_layer = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, curr_layer



class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with a language modeling head.
    This corresponds to HF's GPT2LMHeadModel.
    """

    def __init__(self, config: GPT2Config = GPT2Config(), bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Language Model.
        
        Args:
            config: GPT2Config object containing model configurations.
            bin_path: Path to the pytorch_model.bin file. If empty or None, 
                      weights will not be loaded from file.
        """
        super().__init__()
        
        # TODO: define and initialize the GPT-2 model architecture here. 
        # If the `bin_path` argument is provided, 
            # load the model weights from the specified file path.
        # If `bin_path` is empty or None, do not load any weights, 
            # and initialize the model with random weights.

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)       # token embeddings
        self.wpe = nn.Embedding(config.max_ctx_len, config.d_model)      # position embeddings

        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head — weight-tied with wte
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying!

        # Load checkpoint if provided
        if bin_path:
            state_dict = torch.load(bin_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)



    def forward(
        self, 
        input_ids: Tensor, 
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> CausalLMOutput:
        """
        Forward pass of GPT-2.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            past_key_values: Optional list of past key-value pairs for KV caching

        Returns:
            CausalLMOutput with logits
        """
        # TODO: implement the GPT-2 forward pass here. 
        # The forward pass should compute the output logits for all input tokens,
        # and also update the cached attention keys and values in place (reference passing) 
        # if `past_key_values` is provided.

        B, T = input_ids.shape

        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 0

        # Create position IDs: [0, 1, 2, ..., T-1]
        position_ids = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)  # [1, T]

        # Embeddings
        x = self.wte(input_ids) + self.wpe(position_ids)

        curr_key_values = []

        # Pass through each transformer block
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, curr_layer = block(x, layer_past=layer_past)
            curr_key_values.append(curr_layer)

        if past_key_values is not None:
            for i in range(len(self.h)):
                past_key_values[i] = curr_key_values[i]

        # Final layer norm
        x = self.ln_f(x)

        # LM head to get logits
        logits = self.lm_head(x)   # [B, T, vocab_size]

        return CausalLMOutput(logits=logits)
        
        
    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128
    ) -> ModelOutput:
        """
        Generate tokens autoregressively using KV caching.
        
        Args:
            input_ids: [batch_size, seq_len] starting token IDs
            temperature: Sampling temperature. If 0.0, use greedy sampling.
            top_p: Top-p (nucleus) sampling threshold
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            ModelOutput with `sequences` containing the generated token IDs
        """        
        # TODO: implement the generation method here. 
        # You should use the `forward` method to compute logits and update KV cache at each step.
        # You can assume the input sequences are always padded to the same length,
        # and the total sequence length (input + generated) will not exceed 512 tokens.
        # GPT-2 does not have a stop token,
        # so you should always generate `max_new_tokens` new tokens 
        # for all the input sequences in the batch.

        def sample_token(logits, temperature, top_p):
            if temperature == 0.0:
                return logits.argmax(dim=-1, keepdim=True)

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 0] = False
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_index = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=sampled_index)
            return next_token


        past_key_values = [None] * len(self.h)
        output = self.forward(input_ids, past_key_values=past_key_values)
        next_token_logits = output.logits[:, -1, :]  # [B, vocab_size]

        next_token = sample_token(next_token_logits, temperature, top_p)  # [B, 1]
        generated = input_ids
        generated = torch.cat([generated, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            output = self.forward(next_token, past_key_values=past_key_values)
            next_token_logits = output.logits[:, -1, :]
            next_token = sample_token(next_token_logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)
        
        return ModelOutput(sequences=generated)


class GPT2ForSequenceClassification(nn.Module):
    """
    GPT-2 Model with a classification head.
    """

    def __init__(self, 
                 config: GPT2Config = GPT2Config(), 
                 classifier_bin_path: Optional[str] = None,
                 lm_bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Classification Model.
        
        Args:
            config: GPT2Config object containing model configurations,
                    including the number of labels.
            classifier_bin_path: Path to the 
                    This file should contain the weights for 
                    both the GPT-2 base model and the classification head.
                    If empty or None,
                    the classification head weights will be initialized randomly, 
                    and the base model weights may be initialized randomly 
                    or loaded from `lm_bin_path` if provided.
            lm_bin_path: Path to the pytorch_model.bin file for the language model.
                    This file should contain the weights for the GPT-2 base model.
                    If empty or None,
                    weights may be initialized randomly, 
                    or loaded from `classifier_bin_path` if provided.
        """
        super().__init__()

        # Only one of `classifier_bin_path` and `lm_bin_path` can be provided.
        assert not (classifier_bin_path and lm_bin_path), \
            "Only one of `classifier_bin_path` and `lm_bin_path` can be provided."

        # TODO: define and initialize the GPT-2 model that can be used for sequence classification.
        # You can reuse the GPT2LMHeadModel defined above as the base model,
        # and add a classification head on top of it.
        # You should also reuse GPT2LMHeadModel's weights to speed up training if possible.

        self.config = config
        self.transformer = GPT2LMHeadModel(config)
        self.score = nn.Linear(config.d_model, config.num_labels, bias=False) # classification head instead of vocab-sized word predictor

        if lm_bin_path:
            state_dict = torch.load(lm_bin_path, map_location="cpu")
            self.transformer.load_state_dict(state_dict, strict=False)
        elif classifier_bin_path:
            state_dict = torch.load(classifier_bin_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)


    def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
        """
        Forward pass of GPT-2 for classification.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
        
        Returns:
            SequenceClassifierOutput with logits of shape (batch_size, num_labels)
        """
        
        # TODO: implement the forward pass for sequence classification here.
        # The output logits should be of shape (batch_size, num_labels),
        # where num_labels is specified in the GPT2Config,
        # and the logits contain the classification scores for each label class.

        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)

        for block in self.transformer.h:
            x, _ = block(x)

        hidden_states = self.transformer.ln_f(x)  # [B, T, 768]

        pooled = hidden_states[:, -1, :]  # [B, 768]

        logits = self.score(pooled)  # [B, 20]
        
        return SequenceClassifierOutput(logits=logits)
