import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

class HFWrapperDataset(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): 
        val = self.ds[idx]["input_ids"]
        if not isinstance(val, torch.Tensor): val = torch.tensor(val)
        return val.clone().detach().to(torch.long)

def prepare_data(tokenizer, seq_len=128):
    print("\n[INFO] Processing dataset via disk-cache to save RAM...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)
    
    def tokenize_function(examples):
        texts_with_spaces = [t + " " for t in examples["text"]]
        return tokenizer(texts_with_spaces, add_special_tokens=False)
        
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    def group_texts(examples):
        concatenated = {k: [item for sublist in examples[k] for item in sublist] for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // seq_len) * seq_len
        return {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated.items()
        }
        
    lm_datasets = tokenized.map(group_texts, batched=True, batch_size=1000)
    lm_datasets.set_format(type="torch", columns=["input_ids"])
    
    return HFWrapperDataset(lm_datasets)

def train_baseline(model, dataset, tokenizer, probe_input_ids, n_steps=5000, lr=6e-4,
                   warmup_steps=200, batch_size=64, weight_decay=0.01, grad_clip=1.0, eval_every=250):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    step = 0
    data_iter = iter(dataloader)
    train_tracker = []
    probe_tracker = []

    import time
    start_time = time.time()
    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(DEVICE)
        logits = model(batch)
        if hasattr(logits, 'logits'): logits = logits.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        if step % eval_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                p_out = model(probe_input_ids.to(DEVICE))
                if hasattr(p_out, 'logits'): p_out = p_out.logits
                p_logits = p_out[:, :-1, :].contiguous()
                p_labels = probe_input_ids[:, 1:].to(DEVICE).contiguous()
                probe_losses = F.cross_entropy(
                    p_logits.view(-1, p_logits.size(-1)), p_labels.view(-1), reduction='none'
                ).cpu().numpy()
            model.train()

            train_tracker.append(loss.item())
            probe_tracker.append(probe_losses.mean())
            print(f"     Step {step:>4}/{n_steps} | Train: {loss.item():.3f} | Probe: {probe_losses.mean():.3f}")

    end_time = time.time()
    elapsed = end_time - start_time
    steps_per_sec = n_steps / elapsed

    print(f"\n  [Run Complete] Baseline AdamW Summary:")
    print(f"  Total Time:       {elapsed:.1f} seconds ({steps_per_sec:.2f} steps/sec)")
    print(f"  Final Train Loss: {train_tracker[-1]:.4f} | Final Probe Loss: {probe_tracker[-1]:.4f}")
    print(f"  Avg Train Loss:   {np.mean(train_tracker):.4f} | Avg Probe Loss:   {np.mean(probe_tracker):.4f}\n")

def main():
    print("=" * 72)
    print("  BASELINE: STANDARD ADAMW (No SGT)")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_data(tokenizer)

    print("\n[INFO] Loading Real Validation Data for Robust Probe...")
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    val_texts = [item['text'] for item in val_dataset if len(item['text'].strip()) > 50]
    val_tokens = tokenizer.encode(" ".join(val_texts))

    probe_batch_size = 8
    probe_seq_len = 128
    val_chunks = np.array(val_tokens[:probe_batch_size * probe_seq_len]).reshape(probe_batch_size, probe_seq_len)
    probe_tokens = torch.tensor(val_chunks, dtype=torch.long)

    print("\n[INFO] Initializing Custom ~70.5M LLaMA Architecture from scratch...")
    torch.manual_seed(42)
    llama_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=512, intermediate_size=1376, num_hidden_layers=6, num_attention_heads=8, max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id)
    model = LlamaForCausalLM(llama_config).to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"  -> Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)

    train_baseline(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        probe_input_ids=probe_tokens,
        n_steps=5000,
        lr=6e-4,
        warmup_steps=200,
        batch_size=64,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_every=250
    )

if __name__ == "__main__":
    main()
