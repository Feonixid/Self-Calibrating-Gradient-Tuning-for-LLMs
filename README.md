# Self-Calibrating Gradient Tuning (SGT) for Language Models

SGT (Self-Calibrating Gradient Tuning) is a novel, in-situ training controller designed to dynamically auto-scale token weights and architectural gradients during the training of Large Language Models (LLMs). 

The primary objective of SGT is to introduce closed-loop control theory into the model optimization process. Rather than relying on fixed learning rate schedules or static dataset mixing proportions, SGT continuously monitors the model's loss landscape and allocates gradient capacity precisely to the concepts the model is struggling to learn, while actively preserving previously acquired knowledge.

## Scope of Invention & Authorship
This repository introduces and claims authorship over the following novel mechanisms for training Neural Networks and Large Language Models:

1. **Self-Calibrating Gradient Tuning (SGT)**: The overarching closed-loop architecture for in-situ, proxy-free model optimization.
2. **In-Situ SVD Reference Geometry**: Computing Singular Value Decomposition directly on the immediate loss history and gradient norms inside the primary training loop to determine concept-level learning trajectories, completely eliminating the need for separate proxy models (e.g., DoReMi).
3. **Four-Quadrant Token Routing**: A dynamic sorting mechanism that continuously evaluates the historical exposure and loss state of every vocabulary token, dynamically reallocating gradient capacity into four states:
   * **Drop (Solved)**: Halting gradient compute for mastered concepts.
   * **Retain (Forgetting)**: Aggressively boosting gradients for previously learned concepts that are degrading.
   * **Damp (Volatile)**: Suppressing gradients for high-variance noise.
   * **Scrub (Irreducible)**: Ejecting consistently poor tokens despite massive exposure to prevent dataset poisoning.
4. **Scale-Invariant Signal Extraction**: The use of online Exponential Moving Average (EMA) Z-scoring on the Spatial Coefficient of Variation (CV) and Relative Trend to provide model-agnostic and dataset-agnostic feedback signals.
5. **Closed-Loop Auto-Annealing & Plateau Shocks**: The autonomous interpolation between scratch training and fine-tuning states (via the `Blend` and `Cap` parameters), accompanied by gradient shock injections triggered autonomously by stagnant relative trends.

## Background and Motivation
Modern approaches to token weighting and domain mixing generally rely on a two-phase process. A smaller proxy model must first be trained over the dataset to generate reference losses. The primary model is then trained using these static proxy weights. 

This methodology introduces significant computational overhead and relies on the assumption that a small proxy model's learning dynamics accurately reflect those of a much larger target model.

SGT addresses this limitation by operating entirely online (in-situ). It computes the reference geometry and modulates step sizes directly inside the main model's training loop, relying purely on the model's immediate state. This offers a single-pass, highly efficient alternative to static curriculum learning.

## Experimental Results: WikiText-103 (From Scratch)

The following benchmark demonstrates SGT's capacity to natively scale, allocate resources, and prevent overfitting without manual intervention.

**Configuration:**
* **Dataset:** WikiText-103 (Streaming via HuggingFace Arrow mapping)
* **Architecture:** Pythia-70m (Randomly initialized)
* **Hardware:** 2 GPUs (DataParallel)
* **Training Parameters:** 5,000 steps, Batch Size 64

*Note: Experiments utilizing pre-trained embeddings are currently in progress and will be published in a subsequent update.*

### Full 5,000 Step Benchmark Log (DataParallel)
Below is the complete, raw execution log detailing the step-by-step behavior of the SGT controller on WikiText-103. This log demonstrates the continuous self-calibration of the `Blend`, `Cap`, `Quadrants`, and `Trend` across 5,000 steps, proving that SGT successfully prevented overfitting by dropping over 345 "Solved" tokens and dynamically annealing the controller state.

```text
############################################################
 CONDITION: FROM SCRATCH
############################################################

  -> Using 2 GPUs via DataParallel!
  -> Training Self-Calibrating Multi-Signal SGT v5 (Scale-Invariant)...
       [SGT] Blend: 0.50  | Cap: 5.50 | Q:[Drop=0|Retain=0|Damp=0|Scrub=0] | Trend: +0.0000 | PL: 9.735
       [SGT] Blend: 0.50  | Cap: 5.50 | Q:[Drop=0|Retain=0|Damp=0|Scrub=0] | Trend: +0.0000 | PL: 9.331
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=101|Retain=205|Damp=94|Scrub=0] | Trend: -0.4535 | PL: 8.828
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=47|Retain=212|Damp=82|Scrub=0] | Trend: -0.4334 | PL: 8.458
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=34|Retain=66|Damp=69|Scrub=0] | Trend: -0.4169 | PL: 8.087
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=23|Retain=42|Damp=52|Scrub=0] | Trend: -0.4014 | PL: 7.746
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=17|Retain=26|Damp=46|Scrub=0] | Trend: -0.3715 | PL: 7.454
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=13|Retain=39|Damp=40|Scrub=0] | Trend: -0.3306 | PL: 7.184
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=4|Retain=46|Damp=32|Scrub=0] | Trend: -0.3020 | PL: 6.943
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=5|Retain=52|Damp=29|Scrub=0] | Trend: -0.2696 | PL: 6.735
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=9|Retain=46|Damp=25|Scrub=0] | Trend: -0.2359 | PL: 6.574
       [SGT] Blend: 0.53  | Cap: 7.34 | Q:[Drop=5|Retain=38|Damp=21|Scrub=0] | Trend: -0.2086 | PL: 6.401
       [SGT] Blend: 0.55  | Cap: 7.42 | Q:[Drop=8|Retain=35|Damp=20|Scrub=0] | Trend: -0.1854 | PL: 6.244
       [SGT] Blend: 0.57  | Cap: 7.49 | Q:[Drop=1|Retain=31|Damp=17|Scrub=0] | Trend: -0.1671 | PL: 6.103
     Step  250/5000 | Train: 7.359 | Probe: 5.760
       [SGT] Blend: 0.59  | Cap: 7.55 | Q:[Drop=7|Retain=34|Damp=16|Scrub=0] | Trend: -0.1499 | PL: 5.999
       [SGT] Blend: 0.60  | Cap: 7.60 | Q:[Drop=2|Retain=30|Damp=15|Scrub=0] | Trend: -0.1346 | PL: 5.901
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=1|Retain=28|Damp=15|Scrub=0] | Trend: -0.1145 | PL: 5.826
       [SGT] Blend: 0.62  | Cap: 7.67 | Q:[Drop=2|Retain=23|Damp=15|Scrub=0] | Trend: -0.0924 | PL: 5.783
       [SGT] Blend: 0.63  | Cap: 7.70 | Q:[Drop=2|Retain=17|Damp=13|Scrub=0] | Trend: -0.0726 | PL: 5.740
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=1|Retain=14|Damp=12|Scrub=0] | Trend: -0.0585 | PL: 5.695
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=0|Retain=15|Damp=12|Scrub=0] | Trend: -0.0453 | PL: 5.671
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=1|Retain=13|Damp=9|Scrub=0] | Trend: -0.0364 | PL: 5.648
       [SGT] Blend: 0.64  | Cap: 7.72 | Q:[Drop=1|Retain=9|Damp=9|Scrub=0] | Trend: -0.0283 | PL: 5.645
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=0|Retain=12|Damp=0|Scrub=0] | Trend: -0.0192 | PL: 5.640
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=0|Retain=11|Damp=0|Scrub=0] | Trend: -0.0117 | PL: 5.633
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=0|Retain=9|Damp=0|Scrub=0] | Trend: -0.0079 | PL: 5.626
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=0|Retain=9|Damp=0|Scrub=0] | Trend: -0.0049 | PL: 5.626
       [SGT] Blend: 0.63  | Cap: 7.69 | Q:[Drop=1|Retain=9|Damp=0|Scrub=0] | Trend: -0.0049 | PL: 5.620
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=0|Retain=7|Damp=0|Scrub=0] | Trend: -0.0060 | PL: 5.606
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=2|Retain=7|Damp=2|Scrub=0] | Trend: -0.0069 | PL: 5.597
       [SGT] Blend: 0.64 [SHOCK] | Cap: 7.73 | Q:[Drop=0|Retain=6|Damp=2|Scrub=0] | Trend: -0.0071 | PL: 5.596
       [SGT] Blend: 0.64 [SHOCK] | Cap: 7.76 | Q:[Drop=0|Retain=5|Damp=0|Scrub=0] | Trend: -0.0090 | PL: 5.579
     Step  500/5000 | Train: 5.540 | Probe: 5.491
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=0|Retain=3|Damp=0|Scrub=0] | Trend: -0.0108 | PL: 5.561
       [SGT] Blend: 0.65  | Cap: 7.79 | Q:[Drop=1|Retain=2|Damp=0|Scrub=0] | Trend: -0.0118 | PL: 5.548
       [SGT] Blend: 0.66  | Cap: 7.81 | Q:[Drop=2|Retain=2|Damp=0|Scrub=0] | Trend: -0.0143 | PL: 5.530
       [SGT] Blend: 0.67  | Cap: 7.85 | Q:[Drop=0|Retain=2|Damp=0|Scrub=0] | Trend: -0.0185 | PL: 5.499
       [SGT] Blend: 0.67  | Cap: 7.86 | Q:[Drop=0|Retain=2|Damp=0|Scrub=0] | Trend: -0.0213 | PL: 5.471
       [SGT] Blend: 0.68  | Cap: 7.87 | Q:[Drop=0|Retain=3|Damp=0|Scrub=0] | Trend: -0.0234 | PL: 5.450
       [SGT] Blend: 0.68  | Cap: 7.88 | Q:[Drop=1|Retain=4|Damp=0|Scrub=0] | Trend: -0.0253 | PL: 5.425
       [SGT] Blend: 0.69  | Cap: 7.90 | Q:[Drop=0|Retain=4|Damp=2|Scrub=0] | Trend: -0.0250 | PL: 5.403
       [SGT] Blend: 0.69  | Cap: 7.92 | Q:[Drop=0|Retain=3|Damp=2|Scrub=0] | Trend: -0.0250 | PL: 5.370
       [SGT] Blend: 0.70  | Cap: 7.94 | Q:[Drop=0|Retain=3|Damp=2|Scrub=0] | Trend: -0.0244 | PL: 5.353
       [SGT] Blend: 0.70  | Cap: 7.94 | Q:[Drop=0|Retain=3|Damp=2|Scrub=0] | Trend: -0.0220 | PL: 5.346
       [SGT] Blend: 0.70  | Cap: 7.95 | Q:[Drop=0|Retain=2|Damp=2|Scrub=0] | Trend: -0.0193 | PL: 5.328
       [SGT] Blend: 0.70  | Cap: 7.96 | Q:[Drop=1|Retain=2|Damp=2|Scrub=0] | Trend: -0.0163 | PL: 5.316
       [SGT] Blend: 0.70  | Cap: 7.96 | Q:[Drop=3|Retain=8|Damp=2|Scrub=0] | Trend: -0.0137 | PL: 5.300
       [SGT] Blend: 0.70  | Cap: 7.96 | Q:[Drop=9|Retain=19|Damp=0|Scrub=0] | Trend: -0.0147 | PL: 5.280
       [SGT] Blend: 0.70  | Cap: 7.95 | Q:[Drop=11|Retain=18|Damp=0|Scrub=0] | Trend: -0.0154 | PL: 5.270
       [SGT] Blend: 0.70 [SHOCK] | Cap: 7.96 | Q:[Drop=12|Retain=21|Damp=2|Scrub=0] | Trend: -0.0150 | PL: 5.255
       [SGT] Blend: 0.71 [SHOCK] | Cap: 7.97 | Q:[Drop=21|Retain=26|Damp=2|Scrub=0] | Trend: -0.0161 | PL: 5.232
       [SGT] Blend: 0.71  | Cap: 7.98 | Q:[Drop=35|Retain=31|Damp=2|Scrub=0] | Trend: -0.0190 | PL: 5.198
     Step  750/5000 | Train: 5.305 | Probe: 5.164
       [SGT] Blend: 0.71  | Cap: 7.99 | Q:[Drop=57|Retain=47|Damp=0|Scrub=0] | Trend: -0.0201 | PL: 5.186
       [SGT] Blend: 0.71  | Cap: 7.99 | Q:[Drop=83|Retain=61|Damp=2|Scrub=0] | Trend: -0.0197 | PL: 5.180
       [SGT] Blend: 0.71  | Cap: 7.99 | Q:[Drop=80|Retain=74|Damp=2|Scrub=0] | Trend: -0.0163 | PL: 5.174
       [SGT] Blend: 0.71  | Cap: 7.98 | Q:[Drop=87|Retain=95|Damp=2|Scrub=0] | Trend: -0.0122 | PL: 5.162
       [SGT] Blend: 0.71  | Cap: 8.00 | Q:[Drop=112|Retain=123|Damp=2|Scrub=0] | Trend: -0.0117 | PL: 5.132
       [SGT] Blend: 0.72  | Cap: 8.02 | Q:[Drop=144|Retain=144|Damp=1|Scrub=0] | Trend: -0.0198 | PL: 5.079
       [SGT] Blend: 0.72  | Cap: 8.04 | Q:[Drop=157|Retain=174|Damp=1|Scrub=0] | Trend: -0.0305 | PL: 5.029
       [SGT] Blend: 0.73  | Cap: 8.05 | Q:[Drop=183|Retain=210|Damp=1|Scrub=0] | Trend: -0.0392 | PL: 4.991
       [SGT] Blend: 0.74  | Cap: 8.07 | Q:[Drop=169|Retain=200|Damp=1|Scrub=0] | Trend: -0.0458 | PL: 4.937
       [SGT] Blend: 0.74  | Cap: 8.08 | Q:[Drop=161|Retain=239|Damp=0|Scrub=0] | Trend: -0.0474 | PL: 4.894
       [SGT] Blend: 0.74  | Cap: 8.09 | Q:[Drop=168|Retain=270|Damp=1|Scrub=0] | Trend: -0.0434 | PL: 4.868
       [SGT] Blend: 0.74  | Cap: 8.10 | Q:[Drop=186|Retain=314|Damp=1|Scrub=0] | Trend: -0.0372 | PL: 4.851
       [SGT] Blend: 0.74  | Cap: 8.09 | Q:[Drop=194|Retain=327|Damp=1|Scrub=0] | Trend: -0.0297 | PL: 4.839
       [SGT] Blend: 0.73  | Cap: 8.04 | Q:[Drop=180|Retain=357|Damp=0|Scrub=0] | Trend: -0.0214 | PL: 4.823
       [SGT] Blend: 0.71  | Cap: 7.99 | Q:[Drop=213|Retain=397|Damp=0|Scrub=0] | Trend: -0.0144 | PL: 4.822
       [SGT] Blend: 0.70  | Cap: 7.94 | Q:[Drop=221|Retain=418|Damp=0|Scrub=0] | Trend: -0.0108 | PL: 4.812
       [SGT] Blend: 0.69  | Cap: 7.91 | Q:[Drop=221|Retain=447|Damp=0|Scrub=0] | Trend: -0.0113 | PL: 4.788
       [SGT] Blend: 0.68  | Cap: 7.88 | Q:[Drop=217|Retain=481|Damp=0|Scrub=0] | Trend: -0.0130 | PL: 4.771
       [SGT] Blend: 0.68  | Cap: 7.88 | Q:[Drop=203|Retain=505|Damp=0|Scrub=0] | Trend: -0.0159 | PL: 4.748
       [SGT] Blend: 0.68  | Cap: 7.88 | Q:[Drop=214|Retain=521|Damp=0|Scrub=0] | Trend: -0.0173 | PL: 4.743
     Step 1000/5000 | Train: 4.931 | Probe: 4.682
       [SGT] Blend: 0.68  | Cap: 7.89 | Q:[Drop=211|Retain=545|Damp=0|Scrub=0] | Trend: -0.0171 | PL: 4.725
       [SGT] Blend: 0.68  | Cap: 7.88 | Q:[Drop=223|Retain=571|Damp=0|Scrub=0] | Trend: -0.0146 | PL: 4.715
       [SGT] Blend: 0.68  | Cap: 7.87 | Q:[Drop=233|Retain=601|Damp=0|Scrub=0] | Trend: -0.0142 | PL: 4.695
       [SGT] Blend: 0.68  | Cap: 7.89 | Q:[Drop=233|Retain=615|Damp=1|Scrub=0] | Trend: -0.0158 | PL: 4.668
       [SGT] Blend: 0.69  | Cap: 7.90 | Q:[Drop=224|Retain=652|Damp=1|Scrub=0] | Trend: -0.0189 | PL: 4.648
       [SGT] Blend: 0.69  | Cap: 7.90 | Q:[Drop=210|Retain=665|Damp=1|Scrub=0] | Trend: -0.0201 | PL: 4.630
       [SGT] Blend: 0.69  | Cap: 7.90 | Q:[Drop=215|Retain=685|Damp=1|Scrub=0] | Trend: -0.0212 | PL: 4.610
       [SGT] Blend: 0.69  | Cap: 7.92 | Q:[Drop=217|Retain=702|Damp=1|Scrub=0] | Trend: -0.0207 | PL: 4.589
       [SGT] Blend: 0.69  | Cap: 7.93 | Q:[Drop=241|Retain=715|Damp=1|Scrub=0] | Trend: -0.0215 | PL: 4.557
       [SGT] Blend: 0.70  | Cap: 7.94 | Q:[Drop=247|Retain=726|Damp=1|Scrub=0] | Trend: -0.0213 | PL: 4.547
       [SGT] Blend: 0.69  | Cap: 7.93 | Q:[Drop=249|Retain=734|Damp=1|Scrub=0] | Trend: -0.0198 | PL: 4.535
       [SGT] Blend: 0.70  | Cap: 7.95 | Q:[Drop=250|Retain=752|Damp=1|Scrub=0] | Trend: -0.0163 | PL: 4.530
       [SGT] Blend: 0.69  | Cap: 7.92 | Q:[Drop=215|Retain=752|Damp=1|Scrub=0] | Trend: -0.0113 | PL: 4.528
       [SGT] Blend: 0.68  | Cap: 7.89 | Q:[Drop=206|Retain=756|Damp=1|Scrub=0] | Trend: -0.0053 | PL: 4.532
       [SGT] Blend: 0.67  | Cap: 7.86 | Q:[Drop=220|Retain=772|Damp=2|Scrub=0] | Trend: -0.0024 | PL: 4.532
       [SGT] Blend: 0.66  | Cap: 7.82 | Q:[Drop=224|Retain=772|Damp=2|Scrub=0] | Trend: -0.0007 | PL: 4.528
       [SGT] Blend: 0.66  | Cap: 7.80 | Q:[Drop=220|Retain=771|Damp=2|Scrub=0] | Trend: -0.0006 | PL: 4.526
       [SGT] Blend: 0.66  | Cap: 7.81 | Q:[Drop=213|Retain=777|Damp=2|Scrub=0] | Trend: -0.0027 | PL: 4.514
       [SGT] Blend: 0.66  | Cap: 7.82 | Q:[Drop=221|Retain=784|Damp=2|Scrub=0] | Trend: -0.0073 | PL: 4.493
       [SGT] Blend: 0.67  | Cap: 7.84 | Q:[Drop=221|Retain=783|Damp=1|Scrub=0] | Trend: -0.0120 | PL: 4.472
     Step 1250/5000 | Train: 5.134 | Probe: 4.432
       [SGT] Blend: 0.67  | Cap: 7.85 | Q:[Drop=220|Retain=782|Damp=1|Scrub=0] | Trend: -0.0141 | PL: 4.466
       [SGT] Blend: 0.67  | Cap: 7.85 | Q:[Drop=238|Retain=772|Damp=1|Scrub=0] | Trend: -0.0142 | PL: 4.459
       [SGT] Blend: 0.66  | Cap: 7.82 | Q:[Drop=220|Retain=776|Damp=1|Scrub=0] | Trend: -0.0114 | PL: 4.456
       [SGT] Blend: 0.66  | Cap: 7.79 | Q:[Drop=240|Retain=768|Damp=1|Scrub=0] | Trend: -0.0056 | PL: 4.465
       [SGT] Blend: 0.66  | Cap: 7.79 | Q:[Drop=217|Retain=770|Damp=1|Scrub=0] | Trend: -0.0033 | PL: 4.451
       [SGT] Blend: 0.66  | Cap: 7.80 | Q:[Drop=229|Retain=766|Damp=1|Scrub=0] | Trend: -0.0045 | PL: 4.438
       [SGT] Blend: 0.66  | Cap: 7.80 | Q:[Drop=234|Retain=762|Damp=1|Scrub=0] | Trend: -0.0052 | PL: 4.436
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=245|Retain=754|Damp=1|Scrub=0] | Trend: -0.0061 | PL: 4.433
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=252|Retain=752|Damp=1|Scrub=0] | Trend: -0.0077 | PL: 4.422
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=207|Retain=738|Damp=1|Scrub=0] | Trend: -0.0060 | PL: 4.419
       [SGT] Blend: 0.65  | Cap: 7.77 | Q:[Drop=254|Retain=748|Damp=2|Scrub=0] | Trend: -0.0045 | PL: 4.419
       [SGT] Blend: 0.64  | Cap: 7.76 | Q:[Drop=231|Retain=744|Damp=2|Scrub=0] | Trend: -0.0046 | PL: 4.413
       [SGT] Blend: 0.64  | Cap: 7.74 | Q:[Drop=247|Retain=744|Damp=1|Scrub=0] | Trend: -0.0032 | PL: 4.415
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=219|Retain=740|Damp=1|Scrub=0] | Trend: -0.0031 | PL: 4.404
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=254|Retain=751|Damp=1|Scrub=0] | Trend: -0.0048 | PL: 4.393
       [SGT] Blend: 0.63  | Cap: 7.69 | Q:[Drop=252|Retain=739|Damp=2|Scrub=0] | Trend: -0.0069 | PL: 4.385
       [SGT] Blend: 0.63  | Cap: 7.70 | Q:[Drop=231|Retain=728|Damp=2|Scrub=0] | Trend: -0.0086 | PL: 4.374
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=246|Retain=719|Damp=2|Scrub=0] | Trend: -0.0094 | PL: 4.370
     Step 1500/5000 | Train: 4.726 | Probe: 4.358
       [SGT] Blend: 0.63  | Cap: 7.69 | Q:[Drop=212|Retain=719|Damp=2|Scrub=0] | Trend: -0.0087 | PL: 4.360
       [SGT] Blend: 0.63  | Cap: 7.69 | Q:[Drop=215|Retain=714|Damp=2|Scrub=0] | Trend: -0.0080 | PL: 4.353
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=255|Retain=711|Damp=2|Scrub=0] | Trend: -0.0076 | PL: 4.346
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=238|Retain=708|Damp=2|Scrub=0] | Trend: -0.0049 | PL: 4.356
       [SGT] Blend: 0.61  | Cap: 7.63 | Q:[Drop=209|Retain=696|Damp=2|Scrub=0] | Trend: -0.0026 | PL: 4.355
       [SGT] Blend: 0.61  | Cap: 7.62 | Q:[Drop=237|Retain=696|Damp=1|Scrub=0] | Trend: -0.0002 | PL: 4.355
       [SGT] Blend: 0.60  | Cap: 7.60 | Q:[Drop=224|Retain=693|Damp=1|Scrub=0] | Trend: +0.0002 | PL: 4.349
       [SGT] Blend: 0.60  | Cap: 7.61 | Q:[Drop=233|Retain=694|Damp=1|Scrub=0] | Trend: -0.0016 | PL: 4.338
       [SGT] Blend: 0.61  | Cap: 7.65 | Q:[Drop=211|Retain=680|Damp=1|Scrub=0] | Trend: -0.0062 | PL: 4.324
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=243|Retain=679|Damp=1|Scrub=0] | Trend: -0.0088 | PL: 4.315
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=241|Retain=665|Damp=2|Scrub=0] | Trend: -0.0113 | PL: 4.299
       [SGT] Blend: 0.64  | Cap: 7.75 | Q:[Drop=236|Retain=664|Damp=2|Scrub=0] | Trend: -0.0113 | PL: 4.295
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=245|Retain=664|Damp=2|Scrub=0] | Trend: -0.0089 | PL: 4.297
       [SGT] Blend: 0.66  | Cap: 7.80 | Q:[Drop=227|Retain=655|Damp=2|Scrub=0] | Trend: -0.0058 | PL: 4.295
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=241|Retain=643|Damp=2|Scrub=0] | Trend: -0.0030 | PL: 4.296
       [SGT] Blend: 0.65  | Cap: 7.77 | Q:[Drop=240|Retain=644|Damp=2|Scrub=0] | Trend: -0.0004 | PL: 4.296
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=240|Retain=638|Damp=2|Scrub=0] | Trend: +0.0011 | PL: 4.303
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=256|Retain=634|Damp=2|Scrub=0] | Trend: +0.0009 | PL: 4.298
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=237|Retain=631|Damp=2|Scrub=0] | Trend: -0.0002 | PL: 4.291
     Step 1750/5000 | Train: 4.798 | Probe: 4.268
       [SGT] Blend: 0.64  | Cap: 7.74 | Q:[Drop=243|Retain=624|Damp=2|Scrub=0] | Trend: -0.0048 | PL: 4.267
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=253|Retain=627|Damp=1|Scrub=0] | Trend: -0.0100 | PL: 4.249
       [SGT] Blend: 0.66  | Cap: 7.81 | Q:[Drop=229|Retain=617|Damp=1|Scrub=0] | Trend: -0.0135 | PL: 4.242
       [SGT] Blend: 0.67  | Cap: 7.84 | Q:[Drop=242|Retain=630|Damp=1|Scrub=0] | Trend: -0.0136 | PL: 4.236
       [SGT] Blend: 0.67  | Cap: 7.84 | Q:[Drop=234|Retain=623|Damp=1|Scrub=0] | Trend: -0.0110 | PL: 4.234
       [SGT] Blend: 0.66  | Cap: 7.81 | Q:[Drop=230|Retain=623|Damp=1|Scrub=0] | Trend: -0.0053 | PL: 4.240
       [SGT] Blend: 0.65  | Cap: 7.77 | Q:[Drop=216|Retain=616|Damp=1|Scrub=0] | Trend: -0.0014 | PL: 4.241
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=238|Retain=622|Damp=1|Scrub=0] | Trend: +0.0006 | PL: 4.242
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=242|Retain=602|Damp=1|Scrub=0] | Trend: +0.0002 | PL: 4.232
       [SGT] Blend: 0.64  | Cap: 7.72 | Q:[Drop=261|Retain=603|Damp=1|Scrub=0] | Trend: -0.0022 | PL: 4.223
       [SGT] Blend: 0.64  | Cap: 7.72 | Q:[Drop=251|Retain=603|Damp=1|Scrub=0] | Trend: -0.0065 | PL: 4.208
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=262|Retain=599|Damp=1|Scrub=0] | Trend: -0.0086 | PL: 4.203
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=233|Retain=593|Damp=1|Scrub=0] | Trend: -0.0070 | PL: 4.214
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=236|Retain=598|Damp=1|Scrub=0] | Trend: -0.0040 | PL: 4.211
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=245|Retain=601|Damp=1|Scrub=0] | Trend: -0.0019 | PL: 4.206
       [SGT] Blend: 0.60  | Cap: 7.60 | Q:[Drop=245|Retain=589|Damp=0|Scrub=0] | Trend: -0.0006 | PL: 4.203
       [SGT] Blend: 0.58  | Cap: 7.54 | Q:[Drop=260|Retain=574|Damp=0|Scrub=0] | Trend: -0.0001 | PL: 4.210
       [SGT] Blend: 0.57  | Cap: 7.49 | Q:[Drop=227|Retain=576|Damp=0|Scrub=0] | Trend: +0.0001 | PL: 4.215
     Step 2000/5000 | Train: 4.362 | Probe: 4.227
       [SGT] Blend: 0.55  | Cap: 7.43 | Q:[Drop=232|Retain=568|Damp=0|Scrub=0] | Trend: +0.0023 | PL: 4.220
       [SGT] Blend: 0.53  | Cap: 7.36 | Q:[Drop=234|Retain=554|Damp=0|Scrub=0] | Trend: +0.0049 | PL: 4.229
       [SGT] Blend: 0.51  | Cap: 7.29 | Q:[Drop=251|Retain=560|Damp=0|Scrub=0] | Trend: +0.0070 | PL: 4.239
       [SGT] Blend: 0.50  | Cap: 7.24 | Q:[Drop=225|Retain=553|Damp=0|Scrub=0] | Trend: +0.0068 | PL: 4.241
       [SGT] Blend: 0.48  | Cap: 7.17 | Q:[Drop=216|Retain=555|Damp=0|Scrub=0] | Trend: +0.0067 | PL: 4.247
       [SGT] Blend: 0.47  | Cap: 7.16 | Q:[Drop=235|Retain=563|Damp=0|Scrub=0] | Trend: +0.0043 | PL: 4.239
       [SGT] Blend: 0.46  | Cap: 7.12 | Q:[Drop=230|Retain=565|Damp=0|Scrub=0] | Trend: +0.0028 | PL: 4.247
       [SGT] Blend: 0.48  | Cap: 7.16 | Q:[Drop=226|Retain=554|Damp=0|Scrub=0] | Trend: -0.0007 | PL: 4.233
       [SGT] Blend: 0.49  | Cap: 7.21 | Q:[Drop=235|Retain=553|Damp=0|Scrub=0] | Trend: -0.0023 | PL: 4.232
       [SGT] Blend: 0.50  | Cap: 7.27 | Q:[Drop=237|Retain=553|Damp=0|Scrub=0] | Trend: -0.0044 | PL: 4.223
       [SGT] Blend: 0.52  | Cap: 7.31 | Q:[Drop=252|Retain=552|Damp=0|Scrub=0] | Trend: -0.0049 | PL: 4.219
       [SGT] Blend: 0.52  | Cap: 7.34 | Q:[Drop=242|Retain=545|Damp=0|Scrub=0] | Trend: -0.0063 | PL: 4.213
       [SGT] Blend: 0.53  | Cap: 7.34 | Q:[Drop=256|Retain=541|Damp=0|Scrub=0] | Trend: -0.0045 | PL: 4.213
       [SGT] Blend: 0.54  | Cap: 7.38 | Q:[Drop=251|Retain=543|Damp=0|Scrub=0] | Trend: -0.0043 | PL: 4.209
       [SGT] Blend: 0.55  | Cap: 7.42 | Q:[Drop=244|Retain=551|Damp=0|Scrub=0] | Trend: -0.0043 | PL: 4.200
       [SGT] Blend: 0.56  | Cap: 7.46 | Q:[Drop=248|Retain=537|Damp=0|Scrub=0] | Trend: -0.0043 | PL: 4.198
       [SGT] Blend: 0.57  | Cap: 7.48 | Q:[Drop=264|Retain=534|Damp=0|Scrub=0] | Trend: -0.0039 | PL: 4.197
       [SGT] Blend: 0.58  | Cap: 7.53 | Q:[Drop=250|Retain=537|Damp=0|Scrub=0] | Trend: -0.0050 | PL: 4.186
     Step 2250/5000 | Train: 4.611 | Probe: 4.160
       [SGT] Blend: 0.59  | Cap: 7.57 | Q:[Drop=270|Retain=538|Damp=0|Scrub=0] | Trend: -0.0069 | PL: 4.169
       [SGT] Blend: 0.60  | Cap: 7.61 | Q:[Drop=259|Retain=521|Damp=0|Scrub=0] | Trend: -0.0076 | PL: 4.166
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=233|Retain=521|Damp=0|Scrub=0] | Trend: -0.0082 | PL: 4.162
       [SGT] Blend: 0.62  | Cap: 7.66 | Q:[Drop=274|Retain=515|Damp=0|Scrub=0] | Trend: -0.0069 | PL: 4.163
       [SGT] Blend: 0.62  | Cap: 7.67 | Q:[Drop=264|Retain=523|Damp=0|Scrub=0] | Trend: -0.0042 | PL: 4.161
       [SGT] Blend: 0.62  | Cap: 7.66 | Q:[Drop=284|Retain=514|Damp=0|Scrub=0] | Trend: -0.0019 | PL: 4.159
       [SGT] Blend: 0.62  | Cap: 7.66 | Q:[Drop=252|Retain=525|Damp=0|Scrub=0] | Trend: -0.0020 | PL: 4.155
       [SGT] Blend: 0.62  | Cap: 7.66 | Q:[Drop=272|Retain=515|Damp=0|Scrub=0] | Trend: -0.0028 | PL: 4.148
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=307|Retain=517|Damp=0|Scrub=0] | Trend: -0.0044 | PL: 4.141
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=284|Retain=511|Damp=0|Scrub=0] | Trend: -0.0064 | PL: 4.128
       [SGT] Blend: 0.64  | Cap: 7.74 | Q:[Drop=280|Retain=502|Damp=0|Scrub=0] | Trend: -0.0100 | PL: 4.106
       [SGT] Blend: 0.65  | Cap: 7.78 | Q:[Drop=272|Retain=501|Damp=0|Scrub=0] | Trend: -0.0134 | PL: 4.089
       [SGT] Blend: 0.66  | Cap: 7.82 | Q:[Drop=289|Retain=494|Damp=0|Scrub=0] | Trend: -0.0163 | PL: 4.070
       [SGT] Blend: 0.67  | Cap: 7.85 | Q:[Drop=269|Retain=495|Damp=0|Scrub=0] | Trend: -0.0159 | PL: 4.067
       [SGT] Blend: 0.67  | Cap: 7.85 | Q:[Drop=279|Retain=490|Damp=0|Scrub=0] | Trend: -0.0112 | PL: 4.077
       [SGT] Blend: 0.66  | Cap: 7.79 | Q:[Drop=272|Retain=478|Damp=0|Scrub=0] | Trend: -0.0036 | PL: 4.088
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=286|Retain=487|Damp=0|Scrub=0] | Trend: +0.0036 | PL: 4.101
       [SGT] Blend: 0.62  | Cap: 7.65 | Q:[Drop=292|Retain=485|Damp=0|Scrub=0] | Trend: +0.0081 | PL: 4.104
     Step 2500/5000 | Train: 4.587 | Probe: 4.120
       [SGT] Blend: 0.60  | Cap: 7.61 | Q:[Drop=282|Retain=493|Damp=0|Scrub=0] | Trend: +0.0079 | PL: 4.104
       [SGT] Blend: 0.59  | Cap: 7.57 | Q:[Drop=270|Retain=494|Damp=0|Scrub=0] | Trend: +0.0045 | PL: 4.099
       [SGT] Blend: 0.57  | Cap: 7.50 | Q:[Drop=273|Retain=499|Damp=0|Scrub=0] | Trend: +0.0019 | PL: 4.103
       [SGT] Blend: 0.56  | Cap: 7.45 | Q:[Drop=265|Retain=501|Damp=0|Scrub=0] | Trend: -0.0007 | PL: 4.097
       [SGT] Blend: 0.55  | Cap: 7.43 | Q:[Drop=287|Retain=495|Damp=0|Scrub=0] | Trend: -0.0016 | PL: 4.095
       [SGT] Blend: 0.55  | Cap: 7.41 | Q:[Drop=277|Retain=496|Damp=0|Scrub=0] | Trend: -0.0022 | PL: 4.091
       [SGT] Blend: 0.54  | Cap: 7.40 | Q:[Drop=272|Retain=513|Damp=1|Scrub=0] | Trend: -0.0022 | PL: 4.091
       [SGT] Blend: 0.54  | Cap: 7.40 | Q:[Drop=283|Retain=510|Damp=1|Scrub=0] | Trend: -0.0031 | PL: 4.086
       [SGT] Blend: 0.54  | Cap: 7.40 | Q:[Drop=274|Retain=502|Damp=1|Scrub=0] | Trend: -0.0032 | PL: 4.080
       [SGT] Blend: 0.55  | Cap: 7.42 | Q:[Drop=249|Retain=501|Damp=1|Scrub=0] | Trend: -0.0033 | PL: 4.079
       [SGT] Blend: 0.55  | Cap: 7.43 | Q:[Drop=273|Retain=506|Damp=1|Scrub=0] | Trend: -0.0025 | PL: 4.082
       [SGT] Blend: 0.56  | Cap: 7.44 | Q:[Drop=291|Retain=507|Damp=1|Scrub=0] | Trend: -0.0017 | PL: 4.082
       [SGT] Blend: 0.56  | Cap: 7.46 | Q:[Drop=261|Retain=497|Damp=1|Scrub=0] | Trend: -0.0014 | PL: 4.075
       [SGT] Blend: 0.56  | Cap: 7.48 | Q:[Drop=251|Retain=485|Damp=1|Scrub=0] | Trend: -0.0026 | PL: 4.065
       [SGT] Blend: 0.57  | Cap: 7.51 | Q:[Drop=275|Retain=475|Damp=1|Scrub=0] | Trend: -0.0052 | PL: 4.055
       [SGT] Blend: 0.59  | Cap: 7.55 | Q:[Drop=278|Retain=472|Damp=0|Scrub=0] | Trend: -0.0078 | PL: 4.045
       [SGT] Blend: 0.59  | Cap: 7.57 | Q:[Drop=265|Retain=484|Damp=0|Scrub=0] | Trend: -0.0083 | PL: 4.044
     Step 2750/5000 | Train: 4.238 | Probe: 4.045
       [SGT] Blend: 0.58  | Cap: 7.54 | Q:[Drop=278|Retain=497|Damp=0|Scrub=0] | Trend: -0.0058 | PL: 4.049
       [SGT] Blend: 0.57  | Cap: 7.50 | Q:[Drop=247|Retain=470|Damp=0|Scrub=0] | Trend: -0.0028 | PL: 4.049
       [SGT] Blend: 0.56  | Cap: 7.46 | Q:[Drop=253|Retain=478|Damp=0|Scrub=0] | Trend: -0.0001 | PL: 4.051
       [SGT] Blend: 0.55  | Cap: 7.42 | Q:[Drop=270|Retain=479|Damp=0|Scrub=0] | Trend: +0.0014 | PL: 4.051
       [SGT] Blend: 0.55  | Cap: 7.41 | Q:[Drop=249|Retain=463|Damp=0|Scrub=0] | Trend: +0.0003 | PL: 4.045
       [SGT] Blend: 0.55  | Cap: 7.43 | Q:[Drop=267|Retain=471|Damp=0|Scrub=0] | Trend: -0.0023 | PL: 4.036
       [SGT] Blend: 0.56  | Cap: 7.47 | Q:[Drop=256|Retain=453|Damp=0|Scrub=0] | Trend: -0.0055 | PL: 4.020
       [SGT] Blend: 0.58  | Cap: 7.52 | Q:[Drop=257|Retain=456|Damp=0|Scrub=0] | Trend: -0.0088 | PL: 4.009
       [SGT] Blend: 0.59  | Cap: 7.57 | Q:[Drop=288|Retain=451|Damp=0|Scrub=0] | Trend: -0.0102 | PL: 4.004
       [SGT] Blend: 0.60  | Cap: 7.61 | Q:[Drop=279|Retain=454|Damp=0|Scrub=0] | Trend: -0.0103 | PL: 3.994
       [SGT] Blend: 0.61  | Cap: 7.65 | Q:[Drop=268|Retain=451|Damp=0|Scrub=0] | Trend: -0.0089 | PL: 3.990
       [SGT] Blend: 0.62  | Cap: 7.67 | Q:[Drop=279|Retain=463|Damp=0|Scrub=0] | Trend: -0.0066 | PL: 3.987
       [SGT] Blend: 0.62  | Cap: 7.67 | Q:[Drop=308|Retain=455|Damp=0|Scrub=0] | Trend: -0.0048 | PL: 3.986
       [SGT] Blend: 0.61  | Cap: 7.65 | Q:[Drop=290|Retain=448|Damp=0|Scrub=0] | Trend: -0.0040 | PL: 3.981
       [SGT] Blend: 0.61  | Cap: 7.63 | Q:[Drop=281|Retain=452|Damp=0|Scrub=0] | Trend: -0.0033 | PL: 3.977
       [SGT] Blend: 0.60  | Cap: 7.61 | Q:[Drop=277|Retain=456|Damp=0|Scrub=0] | Trend: -0.0030 | PL: 3.976
       [SGT] Blend: 0.59  | Cap: 7.58 | Q:[Drop=274|Retain=455|Damp=0|Scrub=0] | Trend: -0.0026 | PL: 3.976
       [SGT] Blend: 0.59  | Cap: 7.55 | Q:[Drop=260|Retain=450|Damp=1|Scrub=0] | Trend: -0.0027 | PL: 3.970
       [SGT] Blend: 0.57  | Cap: 7.51 | Q:[Drop=263|Retain=453|Damp=1|Scrub=0] | Trend: -0.0024 | PL: 3.968
     Step 3000/5000 | Train: 4.322 | Probe: 3.963
       [SGT] Blend: 0.57  | Cap: 7.48 | Q:[Drop=273|Retain=452|Damp=1|Scrub=0] | Trend: -0.0028 | PL: 3.964
       [SGT] Blend: 0.56  | Cap: 7.47 | Q:[Drop=253|Retain=448|Damp=2|Scrub=0] | Trend: -0.0038 | PL: 3.958
       [SGT] Blend: 0.57  | Cap: 7.49 | Q:[Drop=266|Retain=443|Damp=2|Scrub=0] | Trend: -0.0046 | PL: 3.952
       [SGT] Blend: 0.57  | Cap: 7.48 | Q:[Drop=265|Retain=435|Damp=2|Scrub=0] | Trend: -0.0044 | PL: 3.950
       [SGT] Blend: 0.57  | Cap: 7.48 | Q:[Drop=276|Retain=433|Damp=2|Scrub=0] | Trend: -0.0039 | PL: 3.950
       [SGT] Blend: 0.55  | Cap: 7.42 | Q:[Drop=271|Retain=430|Damp=2|Scrub=0] | Trend: -0.0021 | PL: 3.954
       [SGT] Blend: 0.53  | Cap: 7.35 | Q:[Drop=292|Retain=434|Damp=2|Scrub=0] | Trend: -0.0001 | PL: 3.956
       [SGT] Blend: 0.51  | Cap: 7.28 | Q:[Drop=288|Retain=448|Damp=1|Scrub=0] | Trend: +0.0012 | PL: 3.957
       [SGT] Blend: 0.49  | Cap: 7.22 | Q:[Drop=296|Retain=446|Damp=1|Scrub=0] | Trend: +0.0013 | PL: 3.955
       [SGT] Blend: 0.48  | Cap: 7.17 | Q:[Drop=275|Retain=434|Damp=1|Scrub=0] | Trend: +0.0012 | PL: 3.958
       [SGT] Blend: 0.46  | Cap: 7.12 | Q:[Drop=302|Retain=433|Damp=1|Scrub=0] | Trend: +0.0010 | PL: 3.960
       [SGT] Blend: 0.45  | Cap: 7.09 | Q:[Drop=284|Retain=429|Damp=1|Scrub=0] | Trend: +0.0010 | PL: 3.960
       [SGT] Blend: 0.44  | Cap: 7.06 | Q:[Drop=245|Retain=411|Damp=1|Scrub=0] | Trend: +0.0010 | PL: 3.960
       [SGT] Blend: 0.44  | Cap: 7.02 | Q:[Drop=265|Retain=424|Damp=1|Scrub=0] | Trend: +0.0013 | PL: 3.962
       [SGT] Blend: 0.43  | Cap: 7.01 | Q:[Drop=296|Retain=421|Damp=1|Scrub=0] | Trend: +0.0006 | PL: 3.961
       [SGT] Blend: 0.45 [SHOCK] | Cap: 7.06 | Q:[Drop=286|Retain=430|Damp=1|Scrub=0] | Trend: +0.0010 | PL: 3.967
       [SGT] Blend: 0.46 [SHOCK] | Cap: 7.11 | Q:[Drop=294|Retain=430|Damp=1|Scrub=0] | Trend: +0.0017 | PL: 3.969
     Step 3250/5000 | Train: 4.316 | Probe: 3.976
       [SGT] Blend: 0.47  | Cap: 7.16 | Q:[Drop=283|Retain=414|Damp=1|Scrub=0] | Trend: +0.0018 | PL: 3.968
       [SGT] Blend: 0.46  | Cap: 7.12 | Q:[Drop=279|Retain=408|Damp=1|Scrub=0] | Trend: +0.0019 | PL: 3.970
       [SGT] Blend: 0.45  | Cap: 7.09 | Q:[Drop=295|Retain=420|Damp=0|Scrub=0] | Trend: +0.0016 | PL: 3.969
       [SGT] Blend: 0.45  | Cap: 7.06 | Q:[Drop=305|Retain=421|Damp=1|Scrub=0] | Trend: +0.0011 | PL: 3.974
       [SGT] Blend: 0.43  | Cap: 7.01 | Q:[Drop=297|Retain=425|Damp=0|Scrub=0] | Trend: +0.0024 | PL: 3.982
       [SGT] Blend: 0.42  | Cap: 6.96 | Q:[Drop=265|Retain=416|Damp=0|Scrub=0] | Trend: +0.0040 | PL: 3.988
       [SGT] Blend: 0.40  | Cap: 6.90 | Q:[Drop=274|Retain=449|Damp=0|Scrub=0] | Trend: +0.0056 | PL: 3.997
       [SGT] Blend: 0.39  | Cap: 6.87 | Q:[Drop=267|Retain=445|Damp=0|Scrub=0] | Trend: +0.0069 | PL: 4.003
       [SGT] Blend: 0.38  | Cap: 6.84 | Q:[Drop=274|Retain=444|Damp=0|Scrub=0] | Trend: +0.0071 | PL: 4.009
       [SGT] Blend: 0.38  | Cap: 6.84 | Q:[Drop=255|Retain=424|Damp=0|Scrub=0] | Trend: +0.0061 | PL: 4.011
       [SGT] Blend: 0.39  | Cap: 6.85 | Q:[Drop=299|Retain=427|Damp=0|Scrub=0] | Trend: +0.0044 | PL: 4.010
       [SGT] Blend: 0.40  | Cap: 6.89 | Q:[Drop=282|Retain=419|Damp=0|Scrub=0] | Trend: +0.0019 | PL: 4.005
       [SGT] Blend: 0.42  | Cap: 6.95 | Q:[Drop=300|Retain=430|Damp=0|Scrub=1] | Trend: -0.0002 | PL: 4.004
       [SGT] Blend: 0.43  | Cap: 7.00 | Q:[Drop=276|Retain=430|Damp=0|Scrub=1] | Trend: -0.0011 | PL: 4.006
     Step 3500/5000 | Train: 4.018 | Probe: 4.009
       [SGT] Blend: 0.44  | Cap: 7.05 | Q:[Drop=289|Retain=420|Damp=0|Scrub=0] | Trend: -0.0012 | PL: 4.005
       [SGT] Blend: 0.45  | Cap: 7.09 | Q:[Drop=307|Retain=411|Damp=0|Scrub=0] | Trend: -0.0007 | PL: 4.005
       [SGT] Blend: 0.46  | Cap: 7.11 | Q:[Drop=309|Retain=425|Damp=0|Scrub=0] | Trend: +0.0004 | PL: 4.008
       [SGT] Blend: 0.46  | Cap: 7.10 | Q:[Drop=302|Retain=440|Damp=0|Scrub=0] | Trend: +0.0012 | PL: 4.011
       [SGT] Blend: 0.45  | Cap: 7.08 | Q:[Drop=282|Retain=436|Damp=0|Scrub=0] | Trend: +0.0020 | PL: 4.016
       [SGT] Blend: 0.45  | Cap: 7.07 | Q:[Drop=302|Retain=436|Damp=0|Scrub=0] | Trend: +0.0028 | PL: 4.017
       [SGT] Blend: 0.45  | Cap: 7.08 | Q:[Drop=298|Retain=430|Damp=0|Scrub=0] | Trend: +0.0022 | PL: 4.014
       [SGT] Blend: 0.46  | Cap: 7.12 | Q:[Drop=276|Retain=433|Damp=0|Scrub=0] | Trend: +0.0006 | PL: 4.010
       [SGT] Blend: 0.48  | Cap: 7.19 | Q:[Drop=280|Retain=435|Damp=0|Scrub=0] | Trend: -0.0014 | PL: 4.005
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=292|Retain=430|Damp=0|Scrub=0] | Trend: -0.0033 | PL: 4.000
       [SGT] Blend: 0.52  | Cap: 7.32 | Q:[Drop=304|Retain=439|Damp=0|Scrub=0] | Trend: -0.0045 | PL: 3.995
       [SGT] Blend: 0.54  | Cap: 7.38 | Q:[Drop=312|Retain=434|Damp=0|Scrub=0] | Trend: -0.0054 | PL: 3.986
       [SGT] Blend: 0.55  | Cap: 7.44 | Q:[Drop=310|Retain=466|Damp=0|Scrub=0] | Trend: -0.0055 | PL: 3.984
       [SGT] Blend: 0.57  | Cap: 7.49 | Q:[Drop=315|Retain=471|Damp=0|Scrub=0] | Trend: -0.0062 | PL: 3.973
       [SGT] Blend: 0.58  | Cap: 7.54 | Q:[Drop=315|Retain=470|Damp=0|Scrub=0] | Trend: -0.0072 | PL: 3.963
       [SGT] Blend: 0.60  | Cap: 7.59 | Q:[Drop=314|Retain=456|Damp=0|Scrub=0] | Trend: -0.0078 | PL: 3.956
       [SGT] Blend: 0.61  | Cap: 7.62 | Q:[Drop=291|Retain=438|Damp=0|Scrub=0] | Trend: -0.0077 | PL: 3.951
     Step 3750/5000 | Train: 4.246 | Probe: 3.942
       [SGT] Blend: 0.61  | Cap: 7.65 | Q:[Drop=333|Retain=442|Damp=0|Scrub=0] | Trend: -0.0064 | PL: 3.954
       [SGT] Blend: 0.61  | Cap: 7.62 | Q:[Drop=306|Retain=441|Damp=0|Scrub=0] | Trend: -0.0031 | PL: 3.958
       [SGT] Blend: 0.59  | Cap: 7.56 | Q:[Drop=321|Retain=443|Damp=0|Scrub=0] | Trend: +0.0006 | PL: 3.966
       [SGT] Blend: 0.57  | Cap: 7.48 | Q:[Drop=284|Retain=428|Damp=0|Scrub=0] | Trend: +0.0039 | PL: 3.973
       [SGT] Blend: 0.54  | Cap: 7.40 | Q:[Drop=304|Retain=432|Damp=0|Scrub=0] | Trend: +0.0061 | PL: 3.980
       [SGT] Blend: 0.53  | Cap: 7.34 | Q:[Drop=277|Retain=417|Damp=0|Scrub=1] | Trend: +0.0061 | PL: 3.981
       [SGT] Blend: 0.51  | Cap: 7.28 | Q:[Drop=290|Retain=419|Damp=0|Scrub=1] | Trend: +0.0047 | PL: 3.981
       [SGT] Blend: 0.50  | Cap: 7.26 | Q:[Drop=323|Retain=434|Damp=0|Scrub=1] | Trend: +0.0027 | PL: 3.980
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=327|Retain=433|Damp=0|Scrub=0] | Trend: +0.0013 | PL: 3.982
       [SGT] Blend: 0.50  | Cap: 7.26 | Q:[Drop=308|Retain=438|Damp=0|Scrub=0] | Trend: +0.0002 | PL: 3.982
       [SGT] Blend: 0.51  | Cap: 7.27 | Q:[Drop=301|Retain=447|Damp=0|Scrub=0] | Trend: -0.0005 | PL: 3.977
       [SGT] Blend: 0.52  | Cap: 7.31 | Q:[Drop=290|Retain=439|Damp=0|Scrub=0] | Trend: -0.0021 | PL: 3.969
       [SGT] Blend: 0.53  | Cap: 7.34 | Q:[Drop=329|Retain=432|Damp=0|Scrub=0] | Trend: -0.0033 | PL: 3.967
       [SGT] Blend: 0.54  | Cap: 7.39 | Q:[Drop=307|Retain=429|Damp=0|Scrub=0] | Trend: -0.0048 | PL: 3.959
       [SGT] Blend: 0.56  | Cap: 7.45 | Q:[Drop=296|Retain=433|Damp=1|Scrub=0] | Trend: -0.0061 | PL: 3.950
       [SGT] Blend: 0.57  | Cap: 7.50 | Q:[Drop=293|Retain=427|Damp=0|Scrub=0] | Trend: -0.0070 | PL: 3.941
       [SGT] Blend: 0.58  | Cap: 7.54 | Q:[Drop=330|Retain=429|Damp=0|Scrub=0] | Trend: -0.0077 | PL: 3.932
     Step 4000/5000 | Train: 4.595 | Probe: 3.915
       [SGT] Blend: 0.60  | Cap: 7.59 | Q:[Drop=298|Retain=424|Damp=0|Scrub=0] | Trend: -0.0081 | PL: 3.928
       [SGT] Blend: 0.61  | Cap: 7.62 | Q:[Drop=281|Retain=431|Damp=0|Scrub=0] | Trend: -0.0077 | PL: 3.920
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=320|Retain=435|Damp=0|Scrub=0] | Trend: -0.0066 | PL: 3.916
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=294|Retain=426|Damp=0|Scrub=0] | Trend: -0.0054 | PL: 3.913
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=321|Retain=439|Damp=0|Scrub=0] | Trend: -0.0052 | PL: 3.905
       [SGT] Blend: 0.62  | Cap: 7.66 | Q:[Drop=299|Retain=437|Damp=0|Scrub=0] | Trend: -0.0058 | PL: 3.897
       [SGT] Blend: 0.62  | Cap: 7.67 | Q:[Drop=314|Retain=420|Damp=0|Scrub=0] | Trend: -0.0063 | PL: 3.889
       [SGT] Blend: 0.63  | Cap: 7.69 | Q:[Drop=300|Retain=412|Damp=0|Scrub=0] | Trend: -0.0073 | PL: 3.882
       [SGT] Blend: 0.63  | Cap: 7.72 | Q:[Drop=313|Retain=410|Damp=0|Scrub=0] | Trend: -0.0075 | PL: 3.876
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=302|Retain=407|Damp=0|Scrub=0] | Trend: -0.0070 | PL: 3.870
       [SGT] Blend: 0.64  | Cap: 7.73 | Q:[Drop=319|Retain=402|Damp=0|Scrub=0] | Trend: -0.0061 | PL: 3.867
       [SGT] Blend: 0.63  | Cap: 7.71 | Q:[Drop=325|Retain=407|Damp=0|Scrub=0] | Trend: -0.0053 | PL: 3.862
       [SGT] Blend: 0.62  | Cap: 7.68 | Q:[Drop=330|Retain=416|Damp=0|Scrub=0] | Trend: -0.0048 | PL: 3.858
       [SGT] Blend: 0.61  | Cap: 7.64 | Q:[Drop=324|Retain=409|Damp=0|Scrub=0] | Trend: -0.0041 | PL: 3.856
       [SGT] Blend: 0.60  | Cap: 7.58 | Q:[Drop=305|Retain=405|Damp=0|Scrub=0] | Trend: -0.0030 | PL: 3.856
       [SGT] Blend: 0.58  | Cap: 7.52 | Q:[Drop=302|Retain=395|Damp=0|Scrub=0] | Trend: -0.0020 | PL: 3.857
       [SGT] Blend: 0.56  | Cap: 7.45 | Q:[Drop=330|Retain=405|Damp=0|Scrub=0] | Trend: -0.0007 | PL: 3.857
       [SGT] Blend: 0.54  | Cap: 7.38 | Q:[Drop=320|Retain=395|Damp=0|Scrub=0] | Trend: +0.0000 | PL: 3.857
     Step 4250/5000 | Train: 4.603 | Probe: 3.846
       [SGT] Blend: 0.52  | Cap: 7.33 | Q:[Drop=337|Retain=405|Damp=1|Scrub=0] | Trend: -0.0004 | PL: 3.853
       [SGT] Blend: 0.51  | Cap: 7.30 | Q:[Drop=345|Retain=401|Damp=1|Scrub=0] | Trend: -0.0017 | PL: 3.847
       [SGT] Blend: 0.51  | Cap: 7.30 | Q:[Drop=327|Retain=412|Damp=1|Scrub=0] | Trend: -0.0030 | PL: 3.843
       [SGT] Blend: 0.52  | Cap: 7.31 | Q:[Drop=336|Retain=404|Damp=1|Scrub=0] | Trend: -0.0033 | PL: 3.843
       [SGT] Blend: 0.52  | Cap: 7.31 | Q:[Drop=341|Retain=402|Damp=1|Scrub=0] | Trend: -0.0028 | PL: 3.844
       [SGT] Blend: 0.51  | Cap: 7.28 | Q:[Drop=333|Retain=396|Damp=2|Scrub=0] | Trend: -0.0018 | PL: 3.842
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=329|Retain=393|Damp=2|Scrub=0] | Trend: -0.0012 | PL: 3.839
       [SGT] Blend: 0.49  | Cap: 7.22 | Q:[Drop=321|Retain=399|Damp=2|Scrub=0] | Trend: -0.0014 | PL: 3.837
       [SGT] Blend: 0.48  | Cap: 7.20 | Q:[Drop=327|Retain=386|Damp=2|Scrub=0] | Trend: -0.0016 | PL: 3.837
       [SGT] Blend: 0.48  | Cap: 7.17 | Q:[Drop=348|Retain=383|Damp=2|Scrub=0] | Trend: -0.0014 | PL: 3.837
       [SGT] Blend: 0.47  | Cap: 7.14 | Q:[Drop=361|Retain=395|Damp=1|Scrub=0] | Trend: -0.0009 | PL: 3.837
       [SGT] Blend: 0.45  | Cap: 7.09 | Q:[Drop=342|Retain=389|Damp=0|Scrub=0] | Trend: -0.0002 | PL: 3.837
       [SGT] Blend: 0.44  | Cap: 7.04 | Q:[Drop=356|Retain=390|Damp=1|Scrub=0] | Trend: +0.0002 | PL: 3.838
       [SGT] Blend: 0.43  | Cap: 7.00 | Q:[Drop=326|Retain=388|Damp=1|Scrub=0] | Trend: +0.0001 | PL: 3.837
       [SGT] Blend: 0.42  | Cap: 6.96 | Q:[Drop=338|Retain=388|Damp=1|Scrub=0] | Trend: +0.0001 | PL: 3.838
       [SGT] Blend: 0.41  | Cap: 6.93 | Q:[Drop=314|Retain=395|Damp=1|Scrub=0] | Trend: +0.0002 | PL: 3.838
     Step 4500/5000 | Train: 4.418 | Probe: 3.839
       [SGT] Blend: 0.40  | Cap: 6.91 | Q:[Drop=315|Retain=390|Damp=1|Scrub=0] | Trend: +0.0002 | PL: 3.838
       [SGT] Blend: 0.39  | Cap: 6.88 | Q:[Drop=322|Retain=401|Damp=0|Scrub=0] | Trend: +0.0003 | PL: 3.839
       [SGT] Blend: 0.39  | Cap: 6.85 | Q:[Drop=323|Retain=405|Damp=1|Scrub=0] | Trend: +0.0005 | PL: 3.840
       [SGT] Blend: 0.40 [SHOCK] | Cap: 6.91 | Q:[Drop=310|Retain=397|Damp=0|Scrub=0] | Trend: +0.0007 | PL: 3.841
       [SGT] Blend: 0.42 [SHOCK] | Cap: 6.97 | Q:[Drop=332|Retain=403|Damp=0|Scrub=0] | Trend: +0.0006 | PL: 3.841
       [SGT] Blend: 0.44  | Cap: 7.03 | Q:[Drop=345|Retain=405|Damp=0|Scrub=0] | Trend: +0.0005 | PL: 3.841
       [SGT] Blend: 0.43  | Cap: 7.01 | Q:[Drop=326|Retain=401|Damp=0|Scrub=0] | Trend: +0.0004 | PL: 3.841
       [SGT] Blend: 0.43  | Cap: 7.01 | Q:[Drop=355|Retain=415|Damp=0|Scrub=0] | Trend: +0.0002 | PL: 3.841
       [SGT] Blend: 0.43  | Cap: 7.01 | Q:[Drop=323|Retain=407|Damp=0|Scrub=0] | Trend: -0.0001 | PL: 3.839
       [SGT] Blend: 0.44  | Cap: 7.04 | Q:[Drop=327|Retain=400|Damp=0|Scrub=0] | Trend: -0.0006 | PL: 3.837
       [SGT] Blend: 0.45  | Cap: 7.08 | Q:[Drop=343|Retain=403|Damp=0|Scrub=0] | Trend: -0.0012 | PL: 3.835
       [SGT] Blend: 0.47  | Cap: 7.13 | Q:[Drop=317|Retain=414|Damp=1|Scrub=0] | Trend: -0.0018 | PL: 3.833
       [SGT] Blend: 0.48  | Cap: 7.18 | Q:[Drop=325|Retain=401|Damp=0|Scrub=0] | Trend: -0.0019 | PL: 3.832
       [SGT] Blend: 0.49  | Cap: 7.22 | Q:[Drop=340|Retain=409|Damp=2|Scrub=0] | Trend: -0.0017 | PL: 3.831
     Step 4750/5000 | Train: 4.395 | Probe: 3.829
       [SGT] Blend: 0.50  | Cap: 7.24 | Q:[Drop=332|Retain=397|Damp=1|Scrub=0] | Trend: -0.0014 | PL: 3.831
       [SGT] Blend: 0.50  | Cap: 7.25 | Q:[Drop=331|Retain=400|Damp=2|Scrub=0] | Trend: -0.0009 | PL: 3.831
       [SGT] Blend: 0.49  | Cap: 7.22 | Q:[Drop=349|Retain=409|Damp=2|Scrub=0] | Trend: -0.0004 | PL: 3.831
       [SGT] Blend: 0.48  | Cap: 7.19 | Q:[Drop=381|Retain=404|Damp=2|Scrub=0] | Trend: +0.0000 | PL: 3.832
       [SGT] Blend: 0.47  | Cap: 7.14 | Q:[Drop=345|Retain=400|Damp=2|Scrub=0] | Trend: +0.0003 | PL: 3.832
       [SGT] Blend: 0.45  | Cap: 7.09 | Q:[Drop=358|Retain=404|Damp=2|Scrub=0] | Trend: +0.0005 | PL: 3.833
       [SGT] Blend: 0.44  | Cap: 7.04 | Q:[Drop=363|Retain=409|Damp=2|Scrub=0] | Trend: +0.0006 | PL: 3.834
       [SGT] Blend: 0.43  | Cap: 7.00 | Q:[Drop=357|Retain=396|Damp=2|Scrub=0] | Trend: +0.0006 | PL: 3.834
       [SGT] Blend: 0.42  | Cap: 6.97 | Q:[Drop=357|Retain=416|Damp=2|Scrub=0] | Trend: +0.0006 | PL: 3.835
       [SGT] Blend: 0.41  | Cap: 6.95 | Q:[Drop=357|Retain=413|Damp=2|Scrub=0] | Trend: +0.0005 | PL: 3.835
       [SGT] Blend: 0.41  | Cap: 6.93 | Q:[Drop=353|Retain=404|Damp=0|Scrub=0] | Trend: +0.0004 | PL: 3.835
       [SGT] Blend: 0.41  | Cap: 6.92 | Q:[Drop=324|Retain=416|Damp=1|Scrub=0] | Trend: +0.0003 | PL: 3.835
       [SGT] Blend: 0.40  | Cap: 6.92 | Q:[Drop=331|Retain=427|Damp=0|Scrub=0] | Trend: +0.0002 | PL: 3.835
       [SGT] Blend: 0.41  | Cap: 6.92 | Q:[Drop=345|Retain=410|Damp=0|Scrub=0] | Trend: +0.0002 | PL: 3.835
     Step 5000/5000 | Train: 4.060 | Probe: 3.836

  Avg Cap:   7.52 | Avg Blend: 0.58 | Avg EMA: 0.62 | Avg Hist: 40 | Avg Freq: 15
  Cap Range: [6.84, 8.10] | Blend Range: [0.38, 0.74]
  Final Train Loss: 4.0600 | Final Probe Loss: 3.8357
  Avg Train Loss:   4.7282 | Avg Probe Loss:   4.2931
```
