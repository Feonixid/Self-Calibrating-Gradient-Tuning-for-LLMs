@ -1,871 +0,0 @@
   

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("sgt_v5_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

                                                                              
                                        
                                                                              

class UniversalSGT:
           

    def __init__(
        self,
        vocab_size,
        probe_tokens,
        n_modes=5,
        use_ema=True,
        total_steps=2000,
        num_layers=None,                                                                      
                                                                                    
        gentle=None,
        aggressive=None,
        layer_ema_gentle=0.95,
        layer_ema_aggressive=0.80,
        layer_weight_range=(0.9, 1.1),
        warmup_decompositions=5,
                                             
        controller_gain=0.06,  
        annealing_strength=0.15,  
        concentration_damp_min=0.7,
        concentration_damp_max=1.3,
        signal_steepness=1.5,
        signal_stat_decay=0.9,
        signal_min_samples=5,
        trend_window=6,
        min_lagging_ratio=1.0,
                                     
        stagnation_patience=10,                                                 
        stagnation_threshold=0.5,                               
        shock_blend_target=0.75,
        shock_duration=3,
        shock_cooldown_cycles=6,                                    
        freq_boost_strength=0.5,
        max_freq_amplification=2.0,
        min_irreducible_observations=20,
        irreducible_exposure_fraction=0.3,
        lr_comp_strength=0.5,
        max_lr_compensation=1.8,
                                     
        solved_sigma=1.5,
        solved_maintenance_weight=0.4,                                                   
        forget_sigma=3.0,
        volatile_sigma=3.0,
        irreducible_sigma=1.5,
        static_frac=0.10,
        volatile_dampen=0.5,
        min_seen_for_quadrants=20,
                                  
        ploss_smooth_decay=0.7,
        vocab_ema_decay=0.7,
    ):
        self.vocab_size = vocab_size
        self.probe_tokens = probe_tokens
        self.n_modes = n_modes
        self.use_ema = use_ema
        self.token_weights = None
        self.probe_token_ids = None
        self.total_steps = total_steps

        self.loss_history = []
        self.step_count = 0

                           
        self.q_solved = 0
        self.q_forget = 0
        self.q_volatile = 0
        self.q_irred = 0

                             
        self.vocab_loss_history_list = []
                                                                              
                                                                               
                                                                                
        self._unseen_loss_value = math.log(vocab_size)
        self.running_vocab_loss = np.full(vocab_size, self._unseen_loss_value, dtype=np.float32)
        self.running_vocab_counts = np.zeros(vocab_size, dtype=np.float32)
        self.seen_mask = np.zeros(vocab_size, dtype=bool)
        self.token_weights = np.ones(vocab_size, dtype=np.float32)

                                           
        self.layer_grad_history_list = []
        if num_layers is None:
            raise ValueError(
                "num_layers must be provided explicitly (e.g. model.config.num_hidden_layers) "
                "- it is no longer assumed/hardcoded."
            )
        self.num_layers = num_layers
        self.layer_weights = np.ones(self.num_layers, dtype=np.float32)
        self.layer_weight_range = layer_weight_range

        self.gentle = gentle or {"cap": 5.5, "ema": 0.85, "hist": 20, "freq": 25, "boost": 2.0}
        self.aggressive = aggressive or {"cap": 9.0, "ema": 0.45, "hist": 55, "freq": 8, "boost": 4.0}
        self.layer_ema_gentle = layer_ema_gentle
        self.layer_ema_aggressive = layer_ema_aggressive

                                                                  
        self.current_dynamic_cap = self.gentle["cap"]
        self.ema_alpha = self.gentle["ema"]
        self.layer_ema_alpha = self.layer_ema_gentle
        self.history_len = self.gentle["hist"]
        self.decompose_every = self.gentle["freq"]
        self.base_boost_factor = self.gentle["boost"]

                                               
        self.cv_history = []
        self.probe_loss_at_decompose = []
        self.smoothed_ploss = None
        self.sv_concentration_history = []

        self.feedback_blend = 0.5
        self.decompose_count = 0
        self.warmup_decompositions = warmup_decompositions

        self.controller_gain = controller_gain
        self.annealing_strength = annealing_strength
        self.concentration_damp_min = concentration_damp_min
        self.concentration_damp_max = concentration_damp_max
        self.signal_steepness = signal_steepness
        self.signal_stat_decay = signal_stat_decay
        self.signal_min_samples = signal_min_samples
        self.trend_window = trend_window
        self.min_lagging_ratio = min_lagging_ratio
        self._signal_stats = {}
        self.forget_sigma = forget_sigma
        self.volatile_sigma = volatile_sigma
        self.irreducible_sigma = irreducible_sigma
        self.static_frac = static_frac
        self.volatile_dampen = volatile_dampen
        self.min_seen_for_quadrants = min_seen_for_quadrants

        self.ploss_smooth_decay = ploss_smooth_decay
        self.vocab_ema_decay = vocab_ema_decay

                              
        self.stagnation_count = 0
        self.shock_active = False
        self.shock_cycles_remaining = 0
        self.shock_cooldown_remaining = 0
        
        self.stagnation_patience = stagnation_patience
        self.stagnation_threshold = stagnation_threshold
        self.shock_blend_target = shock_blend_target
        self.shock_duration = shock_duration
        self.shock_cooldown_cycles = shock_cooldown_cycles
        
        self.freq_boost_strength = freq_boost_strength
        self.max_freq_amplification = max_freq_amplification
        
        self.solved_sigma = solved_sigma
        self.solved_maintenance_weight = solved_maintenance_weight
        
        self.min_irreducible_observations = min_irreducible_observations
        self.irreducible_exposure_fraction = irreducible_exposure_fraction
        
        self.lr_comp_strength = lr_comp_strength
        self.max_lr_compensation = max_lr_compensation
        self.initial_lr = None

                              
        self.cap_tracker = []
        self.ema_tracker = []
        self.hist_tracker = []
        self.freq_tracker = []
        self.blend_tracker = []

    def record_probe_loss(self, model, probe_input_ids):
                                                                      
        model.eval()
        with torch.no_grad():
            logits = model(probe_input_ids.to(DEVICE))
            if hasattr(logits, 'logits'): logits = logits.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = probe_input_ids[:, 1:].to(DEVICE).contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none'
            ).cpu().numpy()
        model.train()
        self.loss_history.append(per_token_loss)
        if len(self.loss_history) > self.history_len:
            self.loss_history.pop(0)

        raw_mean = float(np.mean(per_token_loss))
        if self.smoothed_ploss is None:
            self.smoothed_ploss = raw_mean
        else:
            d = self.ploss_smooth_decay
            self.smoothed_ploss = d * self.smoothed_ploss + (1 - d) * raw_mean

        self.probe_loss_at_decompose.append(self.smoothed_ploss)

    def accumulate_batch(self, per_token_loss_tensor, batch_labels_tensor):
                                                                                         
        losses = per_token_loss_tensor.detach().view(-1).cpu().numpy()
        labels = batch_labels_tensor.detach().view(-1).cpu().numpy()

        counts = np.bincount(labels, minlength=self.vocab_size)
        sums = np.bincount(labels, weights=losses, minlength=self.vocab_size)

        active_mask = counts > 0
        mean_losses = np.zeros(self.vocab_size, dtype=np.float32)
        mean_losses[active_mask] = sums[active_mask] / counts[active_mask]

        d = self.vocab_ema_decay
        for tid in np.where(active_mask)[0]:
            if not self.seen_mask[tid]:
                self.running_vocab_loss[tid] = mean_losses[tid]
                self.seen_mask[tid] = True
            else:
                self.running_vocab_loss[tid] = d * self.running_vocab_loss[tid] + (1 - d) * mean_losses[tid]

        self.running_vocab_counts[active_mask] += counts[active_mask]

                                                                        
                        
                                                                        

    def _compute_probe_trend(self, window=6):
                                                                                                
        losses = self.probe_loss_at_decompose
        if len(losses) < 3:
            return 0.0
        recent = losses[-window:] if len(losses) >= window else losses
        if len(recent) < 3:
            return 0.0
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        n = len(x)
        denom = n * np.sum(x**2) - np.sum(x)**2
        if abs(denom) < 1e-10:
            return 0.0
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denom
        return float(slope)

    def _compute_sv_concentration(self, S):
                                                                              
        if len(S) < 2:
            return 0.5
        total = float(np.sum(S))
        if total < 1e-10:
            return 0.5
        return float(S[0] / total)

    def _update_signal_stats(self, name, value, decay):
                                                        
        stats = self._signal_stats.setdefault(name, {"mean": value, "var": 0.0, "n": 0})
        stats["n"] += 1
        delta = value - stats["mean"]
        stats["mean"] += (1 - decay) * delta
        stats["var"] = decay * stats["var"] + (1 - decay) * delta * delta
        if stats["n"] < self.signal_min_samples:
            return 0.0                                             
        std = math.sqrt(stats["var"]) + 1e-8
        return (value - stats["mean"]) / std

                                                                        
                                               
                                                                        

    def _update_feedback_blend(self):
                   
        progress = min(1.0, self.step_count / max(1, self.total_steps))

        if self.decompose_count <= self.warmup_decompositions:
            self.feedback_blend = 0.5
            return

                                                                                
                                                                                
                                                                          
        probe_trend = self._compute_probe_trend(window=self.trend_window)
        relative_trend = probe_trend / (abs(self.smoothed_ploss) + 1e-6)
        z_trend = self._update_signal_stats("rel_trend", relative_trend, self.signal_stat_decay)
                                                                                             
        trend_target = 1.0 / (1.0 + math.exp(self.signal_steepness * z_trend))

                                                             
                                                                                 
                                                                               
        cv = self.cv_history[-1] if self.cv_history else 0.0
        z_cv = self._update_signal_stats("cv", cv, self.signal_stat_decay)
        cv_target = 1.0 / (1.0 + math.exp(-self.signal_steepness * z_cv))

                                                                             
        target_blend = (1 - progress) * cv_target + progress * trend_target

                                                                                      
                                                                                                                    
        trend_std = math.sqrt(self._signal_stats.get("rel_trend", {"var": 1e-4})["var"]) + 1e-8
        
                                                                                    
        if abs(relative_trend) < (self.stagnation_threshold * trend_std):
            if not getattr(self, 'shock_active', False) and getattr(self, 'shock_cooldown_remaining', 0) <= 0:
                self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            
        if getattr(self, 'shock_cooldown_remaining', 0) > 0:
            self.shock_cooldown_remaining -= 1
            
        if self.stagnation_count >= self.stagnation_patience and not self.shock_active:
            self.shock_active = True
            self.shock_cycles_remaining = self.shock_duration
            self.stagnation_count = 0
            
        if self.shock_active:
            target_blend = self.shock_blend_target
            self.shock_cycles_remaining -= 1
            if self.shock_cycles_remaining <= 0:
                self.shock_active = False
                self.shock_cooldown_remaining = self.shock_cooldown_cycles
                                                

                                                                                 
                                                                        
        sv_concentration = self.sv_concentration_history[-1] if self.sv_concentration_history else 0.5
        concentration_gain = (
            self.concentration_damp_min
            + sv_concentration * (self.concentration_damp_max - self.concentration_damp_min)
        )

        blend_delta = self.controller_gain * (target_blend - self.feedback_blend) * concentration_gain

                                                                                             
        if blend_delta > 0:
            blend_delta *= (1.0 - self.annealing_strength * progress)

        self.feedback_blend = max(0.0, min(1.0, self.feedback_blend + blend_delta))

    def _blend_to_params(self):
                                             
        t = self.feedback_blend
        g, a = self.gentle, self.aggressive

        self.current_dynamic_cap = g["cap"] + t * (a["cap"] - g["cap"])
        self.ema_alpha = g["ema"] + t * (a["ema"] - g["ema"])
        self.base_boost_factor = g["boost"] + t * (a["boost"] - g["boost"])
        self.decompose_every = int(round(g["freq"] + t * (a["freq"] - g["freq"])))

        target_history = int(round(g["hist"] + t * (a["hist"] - g["hist"])))
        if len(self.loss_history) > target_history:
            self.loss_history = self.loss_history[-target_history:]
        self.history_len = target_history

                                                                                  
                                                                     
        self.layer_ema_alpha = self.layer_ema_gentle + t * (self.layer_ema_aggressive - self.layer_ema_gentle)

                                                                        
                                                              
                                                                        

    def decompose_and_reweight(self, model, probe_input_ids, current_lr=None):
        self.vocab_loss_history_list.append(self.running_vocab_loss.copy())
        if len(self.vocab_loss_history_list) > self.history_len:
            self.vocab_loss_history_list.pop(0)

        if len(self.loss_history) < 3:
            return
        self.decompose_count += 1

        if self.probe_token_ids is None:
            self.probe_token_ids = probe_input_ids[:, 1:].contiguous().view(-1).cpu().numpy()

                                                       
        current_losses = self.loss_history[-1]
        cv = float(np.std(current_losses) / (np.mean(current_losses) + 1e-8))
        self.cv_history.append(cv)

                                                                      
                                                
                                                                      
        L = np.array(self.loss_history).T
        L_centered = L - L.mean(axis=1, keepdims=True)
        n_modes = min(self.n_modes, min(L_centered.shape))

        try:
            U, S, Vt = np.linalg.svd(L_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return

        sv_concentration = self._compute_sv_concentration(S[:n_modes])
        self.sv_concentration_history.append(sv_concentration)

                                                                                 
                                                                    
        self._update_feedback_blend()
        self._blend_to_params()

        self.cap_tracker.append(self.current_dynamic_cap)
        self.ema_tracker.append(self.ema_alpha)
        self.hist_tracker.append(self.history_len)
        self.freq_tracker.append(self.decompose_every)
        self.blend_tracker.append(self.feedback_blend)

                                                                                         
                                                                               
                                                                               
                                                                             
                                                                            
                                                                            
        mode_recent_energy = np.zeros(n_modes)
        for i in range(n_modes):
            temporal_profile = Vt[i, :]
            mid = len(temporal_profile) // 2
            if mid > 0:
                recent_energy = np.mean(temporal_profile[mid:] ** 2)
                early_energy = np.mean(temporal_profile[:mid] ** 2) + 1e-10
                mode_recent_energy[i] = recent_energy / early_energy

        lagging_weight = np.maximum(mode_recent_energy - self.min_lagging_ratio, 0.0)
        if lagging_weight.sum() < 1e-8:
                                                                            
                                                                      
            lagging_weight = np.zeros(n_modes)
            lagging_weight[int(np.argmax(mode_recent_energy))] = 1.0
        else:
            lagging_weight = lagging_weight / lagging_weight.sum()

        effective_energy = float(np.sum(lagging_weight * mode_recent_energy))
        boost_val = max(1.0, 1.0 + (effective_energy - 1.0) * self.base_boost_factor)
        
                                                 
        if current_lr is not None and self.initial_lr is not None and self.initial_lr > 0:
            lr_ratio = current_lr / self.initial_lr
            lr_comp = 1.0 + self.lr_comp_strength * (1.0 - lr_ratio)
            lr_comp = min(lr_comp, self.max_lr_compensation)
            boost_val = 1.0 + (boost_val - 1.0) * lr_comp
            
        boost_val = min(self.current_dynamic_cap, boost_val)
                                                 

        combined_loadings = np.zeros(U.shape[0], dtype=np.float64)
        for i in range(n_modes):
            if lagging_weight[i] <= 0:
                continue
            li = np.abs(U[:, i])
            if li.max() > li.min():
                li = (li - li.min()) / (li.max() - li.min())
            else:
                li = np.zeros_like(li)
            combined_loadings += lagging_weight[i] * li

        if combined_loadings.max() > combined_loadings.min():
            normalized = (combined_loadings - combined_loadings.min()) / (combined_loadings.max() - combined_loadings.min())
        else:
            normalized = np.zeros_like(combined_loadings)

        probe_weights = 1.0 + normalized * (boost_val - 1.0)

                                                       
        vocab_weights_dict = {}
        probe_ids = self.probe_token_ids
        for pos, weight in enumerate(probe_weights):
            if pos < len(probe_ids):
                tid = int(probe_ids[pos])
                vocab_weights_dict.setdefault(tid, []).append(weight)
        vocab_weights_dict = {tid: np.mean(w) for tid, w in vocab_weights_dict.items()}

        new_weights = np.ones(self.vocab_size, dtype=np.float32)
        for tid, w in vocab_weights_dict.items():
            new_weights[tid] = w

                                                          
        seen_idx_all = np.where(self.seen_mask)[0]
        if len(seen_idx_all) > 0 and np.median(self.running_vocab_counts[seen_idx_all]) > 10:
            median_count = np.median(self.running_vocab_counts[seen_idx_all])
            counts = self.running_vocab_counts
            exposure_ratio = counts / (median_count + 1)
            freq_factor = 1.0 + self.freq_boost_strength * np.maximum(0, 1.0 - exposure_ratio)
            freq_factor = np.minimum(freq_factor, self.max_freq_amplification)
            
            boost_mask = new_weights > 1.0
            new_weights[boost_mask] = 1.0 + (new_weights[boost_mask] - 1.0) * freq_factor[boost_mask]
            new_weights = np.minimum(new_weights, self.current_dynamic_cap)
                                                          

                                                                      
                                                       
                                                                      
        seen_idx = np.where(self.seen_mask)[0]
        if len(seen_idx) >= self.min_seen_for_quadrants and len(self.vocab_loss_history_list) >= 3:
                                                                              
                                                                                  
                                                                                  
                                                                                  
                                                
            L_vocab_full = np.array(self.vocab_loss_history_list).T
            L_vocab = L_vocab_full[seen_idx, :]
            velocities = np.mean(np.diff(L_vocab, axis=1), axis=1)
            mean_losses = L_vocab.mean(axis=1)
            volatility = np.var(L_vocab, axis=1)

            mean_vel, std_vel = float(np.mean(velocities)), float(np.std(velocities))
            mean_vol, std_vol = float(np.mean(volatility)), float(np.std(volatility))
            vocab_mean_loss, vocab_std_loss = float(np.mean(mean_losses)), float(np.std(mean_losses))
            
            progress = min(1.0, self.step_count / max(1, self.total_steps))

                                                                
            solved_threshold = vocab_mean_loss - self.solved_sigma * vocab_std_loss
            is_static = np.abs(velocities) < (self.static_frac * std_vel)
            solved = (mean_losses < solved_threshold) & is_static

                                                        
                                                                    
            forget_threshold = max(0.0, mean_vel + self.forget_sigma * std_vel)
            is_forgetting = velocities > forget_threshold
            
            if len(self.vocab_loss_history_list) >= 4:
                                                           
                L_vocab_4 = np.array(self.vocab_loss_history_list[-4:]).T[seen_idx, :]
                v1 = np.mean(np.diff(L_vocab_4[:, :-1], axis=1), axis=1)
                v2 = np.mean(np.diff(L_vocab_4[:, 1:], axis=1), axis=1)
                accel = v2 - v1
                
                accelerating = is_forgetting & (accel > 0)
                decelerating = is_forgetting & (accel <= 0)
            else:
                accelerating = is_forgetting
                decelerating = np.zeros(len(seen_idx), dtype=bool)

                         
            volatile_threshold = mean_vol + self.volatile_sigma * std_vol
            is_volatile = volatility > volatile_threshold

                                                           
            median_count = np.median(self.running_vocab_counts[seen_idx])
            min_exp = max(self.min_irreducible_observations, median_count * self.irreducible_exposure_fraction)
            has_sufficient_exposure = self.running_vocab_counts[seen_idx] >= min_exp
            
            high_loss_threshold = vocab_mean_loss + self.irreducible_sigma * vocab_std_loss
            is_high_loss = mean_losses > high_loss_threshold
            is_static = np.abs(velocities) < (self.static_frac * std_vel)
            irreducible = is_high_loss & is_static & has_sufficient_exposure

                                                  
            solved_mask = np.zeros(self.vocab_size, dtype=bool); solved_mask[seen_idx] = solved
            accel_mask = np.zeros(self.vocab_size, dtype=bool); accel_mask[seen_idx] = accelerating
            decel_mask = np.zeros(self.vocab_size, dtype=bool); decel_mask[seen_idx] = decelerating
            is_forgetting_full = accel_mask | decel_mask
            
            is_volatile_full = np.zeros(self.vocab_size, dtype=bool); is_volatile_full[seen_idx] = is_volatile
            irreducible_mask = np.zeros(self.vocab_size, dtype=bool); irreducible_mask[seen_idx] = irreducible

                                                                             
                                                                                      
            new_weights[irreducible_mask] = 1.0 + (new_weights[irreducible_mask] - 1.0) * 0.5
            
                                                                               
            new_weights[solved_mask] *= self.solved_maintenance_weight
            
                                                                                                 
            new_weights[accel_mask] = np.maximum(new_weights[accel_mask], boost_val * 0.8)
            
                                                       
            new_weights[decel_mask] = np.maximum(new_weights[decel_mask], 1.0 + (boost_val - 1.0) * 0.5)
            
                                                       
            new_weights = np.minimum(new_weights, self.current_dynamic_cap)

            volatile_boosts = is_volatile_full & (new_weights > 1.0) & (~is_forgetting_full)
            new_weights[volatile_boosts] = 1.0 + (new_weights[volatile_boosts] - 1.0) * self.volatile_dampen

            self.q_solved = int(np.sum(solved_mask))
            self.q_forget = int(np.sum(is_forgetting_full))
            self.q_volatile = int(np.sum(volatile_boosts))
            self.q_irred = int(np.sum(irreducible_mask))
        else:
                                                                                 
                                                                                  
                                                                        
            self.q_solved = self.q_forget = self.q_volatile = self.q_irred = 0

        if self.use_ema and self.token_weights is not None:
            self.token_weights = self.ema_alpha * self.token_weights + (1.0 - self.ema_alpha) * new_weights
        else:
            self.token_weights = new_weights

                                                                        
                                                  
                                                                        

    def decompose_layers(self):
                                                                                                
        if len(self.layer_grad_history_list) < 3:
            return

        L_grad = np.array(self.layer_grad_history_list).T
        L_grad_centered = L_grad - L_grad.mean(axis=1, keepdims=True)

        try:
            U_g, S_g, Vt_g = np.linalg.svd(L_grad_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return

        n_modes = min(3, min(L_grad_centered.shape))
        mode_recent_energy = np.zeros(n_modes)
        for i in range(n_modes):
            temporal_profile = Vt_g[i, :]
            mid = len(temporal_profile) // 2
            if mid > 0:
                recent_energy = np.mean(temporal_profile[mid:] ** 2)
                early_energy = np.mean(temporal_profile[:mid] ** 2) + 1e-10
                mode_recent_energy[i] = recent_energy / early_energy

        lagging_mode = int(np.argmax(mode_recent_energy))
        layer_loadings = np.abs(U_g[:, lagging_mode])

        if layer_loadings.max() > layer_loadings.min():
            normalized = (layer_loadings - layer_loadings.min()) / (layer_loadings.max() - layer_loadings.min())
        else:
            normalized = np.zeros_like(layer_loadings)

        lo, hi = self.layer_weight_range
        raw_layer_weights = lo + normalized * (hi - lo)

                                                                               
                                                  
        smoothed_weights = np.copy(raw_layer_weights)
        if len(raw_layer_weights) >= 3:
            for i in range(1, len(raw_layer_weights) - 1):
                smoothed_weights[i] = 0.25 * raw_layer_weights[i - 1] + 0.50 * raw_layer_weights[i] + 0.25 * raw_layer_weights[i + 1]

                                                                                 
        self.layer_weights = self.layer_ema_alpha * self.layer_weights + (1.0 - self.layer_ema_alpha) * smoothed_weights

    def record_and_apply_layer_grads(self, model):
                                                                          
        actual_model = model.module if hasattr(model, 'module') else model
        if self.step_count % self.decompose_every == 0:
            grad_norms = []
            for layer in actual_model.model.layers:
                total_norm = 0.0
                for p in layer.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.detach().data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)

            self.layer_grad_history_list.append(np.array(grad_norms))
            if len(self.layer_grad_history_list) > self.history_len:
                self.layer_grad_history_list.pop(0)
            self.decompose_layers()

        for i, layer in enumerate(actual_model.model.layers):
            weight = float(self.layer_weights[i])
            if abs(weight - 1.0) > 1e-4:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(weight)

    def get_weighted_loss(self, per_token_loss, batch_labels, model):
        labels_flat = batch_labels.view(-1).cpu().numpy()
        w_array = self.token_weights[labels_flat]
        weights = torch.tensor(w_array, dtype=per_token_loss.dtype, device=DEVICE)
        return (per_token_loss * weights).mean()

    def step(self, model, probe_input_ids, per_token_loss, batch_labels, current_lr=None):
        if current_lr is not None and self.initial_lr is None:
            self.initial_lr = current_lr
            
        self.step_count += 1
        self.accumulate_batch(per_token_loss, batch_labels)

        if self.step_count % self.decompose_every == 0:
            self.record_probe_loss(model, probe_input_ids)
            self.decompose_and_reweight(model, probe_input_ids, current_lr)

            cv = self.cv_history[-1] if self.cv_history else 0
            trend = self._compute_probe_trend(window=self.trend_window)
            ploss = self.probe_loss_at_decompose[-1] if self.probe_loss_at_decompose else 0
            shock_str = "[SHOCK]" if getattr(self, 'shock_active', False) else ""
            print(f"       [SGT] Blend: {self.feedback_blend:.2f} {shock_str} | Cap: {self.current_dynamic_cap:.2f} | "
                  f"Q:[Drop={self.q_solved}|Retain={self.q_forget}|Damp={self.q_volatile}|Scrub={self.q_irred}] | "
                  f"Trend: {trend:+.4f} | PL: {ploss:.3f}")

                                                                              
                      
                                                                              

class HFWrapperDataset(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): 
        val = self.ds[idx]["input_ids"]
        if not isinstance(val, torch.Tensor): val = torch.tensor(val)
        return val.clone().detach().to(torch.long)

def prepare_data(tokenizer, seq_len=128):
    print("\n Processing dataset...")
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

def train_auto_scaling(model, dataset, tokenizer, probe_input_ids, n_steps=1000, lr=3e-4,
                        warmup_steps=200, batch_size=16, weight_decay=0.01, grad_clip=1.0, eval_every=250,
                        sgt_kwargs=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

                                                                               
                                                                                          
    actual_model = model.module if hasattr(model, 'module') else model
    sgt = UniversalSGT(
        vocab_size=len(tokenizer),
        probe_tokens=probe_input_ids,
        total_steps=n_steps,
        num_layers=actual_model.config.num_hidden_layers,
        **(sgt_kwargs or {}),
    )

    step = 0
    data_iter = iter(dataloader)

    train_tracker = []
    probe_tracker = []

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

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none'
        )

        loss = sgt.get_weighted_loss(per_token_loss, shift_labels, model)
        current_lr = scheduler.get_last_lr()[0]
        sgt.step(model, probe_input_ids, per_token_loss, shift_labels, current_lr)

        optimizer.zero_grad()
        loss.backward()

        sgt.record_and_apply_layer_grads(model)

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

    print(f"\n Run Summary:")
    if sgt.cap_tracker:
        print(f"  Avg Cap:   {np.mean(sgt.cap_tracker):.2f} | Avg Blend: {np.mean(sgt.blend_tracker):.2f} | "
              f"Avg EMA: {np.mean(sgt.ema_tracker):.2f} | Avg Hist: {np.mean(sgt.hist_tracker):.0f} | "
              f"Avg Freq: {np.mean(sgt.freq_tracker):.0f}")
        print(f"  Cap Range: [{min(sgt.cap_tracker):.2f}, {max(sgt.cap_tracker):.2f}] | "
              f"Blend Range: [{min(sgt.blend_tracker):.2f}, {max(sgt.blend_tracker):.2f}]")
    print(f"  Final Train Loss: {train_tracker[-1]:.4f} | Final Probe Loss: {probe_tracker[-1]:.4f}")
    print(f"  Avg Train Loss:   {np.mean(train_tracker):.4f} | Avg Probe Loss:   {np.mean(probe_tracker):.4f}\n")

def main():
    print("=" * 72)
    print(" SGT v5")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_data(tokenizer)

    print("\nFetching Validation Data")
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    val_texts = [item['text'] for item in val_dataset if len(item['text'].strip()) > 50]
    val_tokens = tokenizer.encode(" ".join(val_texts))

    probe_batch_size = 8
    probe_seq_len = 128
    val_chunks = np.array(val_tokens[:probe_batch_size * probe_seq_len]).reshape(probe_batch_size, probe_seq_len)
    probe_tokens = torch.tensor(val_chunks, dtype=torch.long)

    print("\n[INFO] Fetching 512D Pre-Trained Embeddings from Pythia-70m...")
    pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    pretrained_embeds = pythia_model.gpt_neox.embed_in.weight.data[:len(tokenizer), :].clone()
    del pythia_model

    CONDITIONS = [
        {"name": "FROM SCRATCH", "use_pretrained": False},
        {"name": "PRE-TRAINED EMBEDDINGS", "use_pretrained": True}
    ]

    for cond in CONDITIONS:
        cond_name = cond["name"]
        print(f"\n\n{'#'*60}\n CONDITION: {cond_name}\n{'#'*60}\n")

        torch.manual_seed(42)
        llama_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=512, intermediate_size=1376, num_hidden_layers=6, num_attention_heads=8, max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id)
        model = LlamaForCausalLM(llama_config).to(DEVICE)

        if cond["use_pretrained"]:
            model.model.embed_tokens.weight.data.copy_(pretrained_embeds.to(DEVICE))
            model.model.embed_tokens.weight.requires_grad = False

        if torch.cuda.device_count() > 1:
            print(f"  -> Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        print("  -> Training with SGT v5")
        train_auto_scaling(model, dataset, tokenizer, probe_tokens, n_steps=5000, lr=6e-4, batch_size=64, warmup_steps=200)

if __name__ == "__main__": main()