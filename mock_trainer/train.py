import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

print("\n[TRAINER ENGINE] Initializing PyTorch PEFT/LoRA on CPU...")

# 1. Load a tiny dummy model suitable for CPU testing
model_id = "sshleifer/tiny-gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. Configure LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # Target GPT-2 attention blocks
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Wrap the model in the LoRA adapter (Freezes base weights, trains only adapter)
model = get_peft_model(model, peft_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("[TRAINER ENGINE] LoRA Adapter Successfully Injected.")
model.print_trainable_parameters()

# Dummy input to simulate the agent's interaction data
dummy_input = torch.randint(0, 50257, (1, 10))

print("[TRAINER ENGINE] Entering Asynchronous Training Loop...")
while True:
    time.sleep(10)  # Simulate waiting for the PRM to batch next-state rewards

    print("\n[TRAINER ENGINE] Pulling reward batch from PRM...")

    # 4. Real PyTorch Training Step
    optimizer.zero_grad()

    # Forward pass
    outputs = model(dummy_input, labels=dummy_input)
    loss = outputs.loss

    # Backward pass (calculating gradients mathematically)
    loss.backward()

    # Update weights
    optimizer.step()

    print(
        f"[TRAINER ENGINE] Backpropagation Complete. Loss: {loss.item():.4f} | Weights Updated."
    )
