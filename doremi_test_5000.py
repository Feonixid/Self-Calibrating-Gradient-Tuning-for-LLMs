import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import get_cosine_schedule_with_warmup
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
    print("\nProcessing dataset")
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

def train_baseline_doremi(model, proxy_model, dataset, tokenizer, probe_input_ids, n_steps=5000, proxy_steps=1000, lr=6e-4,
                   warmup_steps=200, batch_size=64, weight_decay=0.01, grad_clip=1.0, eval_every=250):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    start_time = time.time()

    print(f"\nPre-training Proxy Model for {proxy_steps} steps")
    proxy_model.train()
    proxy_optimizer = torch.optim.AdamW(proxy_model.parameters(), lr=lr, weight_decay=weight_decay)
    proxy_scheduler = get_cosine_schedule_with_warmup(proxy_optimizer, num_warmup_steps=100, num_training_steps=proxy_steps)
    
    proxy_step = 0
    while proxy_step < proxy_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(DEVICE)
        logits = proxy_model(batch)
        if hasattr(logits, 'logits'): logits = logits.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        proxy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy_model.parameters(), grad_clip)
        proxy_optimizer.step()
        proxy_scheduler.step()
        
        proxy_step += 1
        if proxy_step % 250 == 0 or proxy_step == proxy_steps:
            print(f"     Proxy Pre-train Step {proxy_step:>4}/{proxy_steps} | Loss: {loss.item():.3f}")

    proxy_model.eval()
    for param in proxy_model.parameters():
        param.requires_grad = False
    del proxy_optimizer, proxy_scheduler
    torch.cuda.empty_cache()

    print(f"\nTraining Main with Proxy Weights for {n_steps} steps")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps)

    step = 0
    train_tracker = []
    probe_tracker = []
    
    data_iter = iter(dataloader)
    
    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(DEVICE)

        shift_labels = batch[:, 1:].contiguous()
        flat_labels = shift_labels.reshape(-1)

        with torch.no_grad():
            proxy_out = proxy_model(batch)
            if hasattr(proxy_out, 'logits'): proxy_out = proxy_out.logits
            p_shift = proxy_out[:, :-1, :].contiguous()
            proxy_loss = F.cross_entropy(
                p_shift.reshape(-1, p_shift.size(-1)), flat_labels, reduction='none'
            )
            # Detach and keep on same device; free the large logit tensor immediately
            proxy_loss = proxy_loss.detach()
            del proxy_out, p_shift

        main_out = model(batch)
        if hasattr(main_out, 'logits'): main_out = main_out.logits
        m_shift = main_out[:, :-1, :].contiguous()
        main_loss = F.cross_entropy(
            m_shift.reshape(-1, m_shift.size(-1)), flat_labels, reduction='none'
        )

        # ── DoReMi re-weighting ──
        excess_loss = main_loss - proxy_loss
        weights = torch.exp(torch.clamp(excess_loss, -5.0, 5.0))
        weights = weights / (weights.mean() + 1e-8) 

        loss = (main_loss * weights.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # Free intermediates every step to keep T4 headroom
        del main_out, m_shift, proxy_loss, excess_loss, weights

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

            train_tracker.append(main_loss.mean().item())
            probe_tracker.append(probe_losses.mean())
            print(f"     Step {step:>4}/{n_steps} | Train (Unweighted): {main_loss.mean().item():.3f} | Probe: {probe_losses.mean():.3f}")

    end_time = time.time()
    elapsed = end_time - start_time
    total_steps = proxy_steps + n_steps
    steps_per_sec = total_steps / elapsed

    print(f"\n  [Run Complete] Baseline DoReMi Summary:")
    print(f"  Total Training Time (proxy + main): {elapsed:.1f} seconds ({steps_per_sec:.2f} steps/sec)")
    print(f"  Steps breakdown: {proxy_steps} proxy + {n_steps} main = {total_steps} total")
    print(f"  Final Train Loss: {train_tracker[-1]:.4f} | Final Probe Loss: {probe_tracker[-1]:.4f}")
    print(f"  Avg Train Loss:   {np.mean(train_tracker):.4f} | Avg Probe Loss:   {np.mean(probe_tracker):.4f}\n")

def main():
    print("=" * 72)
    print("  BASELINE: DoReMi (Proxy pre-trained, then Main Model)")
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
    
    
    print("Initializing 2-Layer Proxy Model for Reference")
    proxy_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id)
    proxy_model = LlamaForCausalLM(proxy_config).to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"  -> Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        proxy_model = nn.DataParallel(proxy_model)

    train_baseline_doremi(
        model=model,
        proxy_model=proxy_model,
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
