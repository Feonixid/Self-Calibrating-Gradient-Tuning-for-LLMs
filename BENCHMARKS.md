# Self-Calibrating Gradient Tuning - Benchmarks


## Summary Table

| Optimizer / Strategy | Total Time (s) | Steps/sec | Final Train Loss | Final Probe Loss | Avg Train Loss | Avg Probe Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 3549.3 | 1.41 | 4.0753 | 4.0390 | 4.4961 | 4.4912 |
| **Adafactor** | 3589.4 | 1.39 | 4.2020 | 4.2328 | 4.6439 | 4.7042 |
| **DoReMi** | 4897.9 | 1.23 | 4.5236 | 4.7227 | 4.9385 | 5.0284 |
| **RMSprop** (1e-3) | 3460.2 | 1.45 | 6.6607 | 6.5813 | 6.8215 | 6.8378 |
| **RMSprop** (1e-4) | 3644.2 | 1.37 | 7.2607 | 7.0949 | 7.1143 | 7.0506 |
| **SGD Momentum** (lr 0.01) | 3478.8 | 1.44 | 6.7849 | 6.7465 | 6.9457 | 6.9792 |
| **SGD Momentum** (lr 0.1) | 3660.8 | 1.37 | 6.6625 | 6.6204 | 6.8367 | 6.8808 |

---

## Detailed Step Logs

### AdamW
```text
Step  250/5000 | Train: 6.123 | Probe: 6.147
Step  500/5000 | Train: 5.514 | Probe: 5.488
Step  750/5000 | Train: 5.168 | Probe: 5.254
Step 1000/5000 | Train: 4.868 | Probe: 4.916
Step 1250/5000 | Train: 4.980 | Probe: 4.774
Step 1500/5000 | Train: 4.582 | Probe: 4.606
Step 1750/5000 | Train: 4.521 | Probe: 4.503
Step 2000/5000 | Train: 4.390 | Probe: 4.403
Step 2250/5000 | Train: 4.285 | Probe: 4.383
Step 2500/5000 | Train: 4.374 | Probe: 4.350
Step 2750/5000 | Train: 4.266 | Probe: 4.212
Step 3000/5000 | Train: 4.156 | Probe: 4.183
Step 3250/5000 | Train: 4.202 | Probe: 4.169
Step 3500/5000 | Train: 4.087 | Probe: 4.118
Step 3750/5000 | Train: 4.196 | Probe: 4.065
Step 4000/5000 | Train: 4.085 | Probe: 4.082
Step 4250/5000 | Train: 3.890 | Probe: 4.052
Step 4500/5000 | Train: 4.216 | Probe: 4.040
Step 4750/5000 | Train: 3.941 | Probe: 4.039
Step 5000/5000 | Train: 4.075 | Probe: 4.039
```

### Adafactor
```text
Step  250/5000 | Train: 6.365 | Probe: 6.402
Step  500/5000 | Train: 5.714 | Probe: 5.771
Step  750/5000 | Train: 5.363 | Probe: 5.541
Step 1000/5000 | Train: 5.041 | Probe: 5.192
Step 1250/5000 | Train: 5.127 | Probe: 4.990
Step 1500/5000 | Train: 4.737 | Probe: 4.828
Step 1750/5000 | Train: 4.676 | Probe: 4.719
Step 2000/5000 | Train: 4.544 | Probe: 4.595
Step 2250/5000 | Train: 4.449 | Probe: 4.576
Step 2500/5000 | Train: 4.517 | Probe: 4.522
Step 2750/5000 | Train: 4.398 | Probe: 4.407
Step 3000/5000 | Train: 4.293 | Probe: 4.377
Step 3250/5000 | Train: 4.334 | Probe: 4.372
Step 3500/5000 | Train: 4.203 | Probe: 4.297
Step 3750/5000 | Train: 4.332 | Probe: 4.263
Step 4000/5000 | Train: 4.200 | Probe: 4.275
Step 4250/5000 | Train: 3.997 | Probe: 4.253
Step 4500/5000 | Train: 4.345 | Probe: 4.239
Step 4750/5000 | Train: 4.040 | Probe: 4.232
Step 5000/5000 | Train: 4.202 | Probe: 4.233
```

### DoReMi
```text
Proxy Pre-train Step  250/1000 | Loss: 6.239
Proxy Pre-train Step  500/1000 | Loss: 5.580
Proxy Pre-train Step  750/1000 | Loss: 5.371
Proxy Pre-train Step 1000/1000 | Loss: 5.358

Training Main with Proxy Weights for 5000 steps
Step  250/5000 | Train (Unweighted): 6.243 | Probe: 6.347
Step  500/5000 | Train (Unweighted): 5.538 | Probe: 5.648
Step  750/5000 | Train (Unweighted): 5.446 | Probe: 5.394
Step 1000/5000 | Train (Unweighted): 5.176 | Probe: 5.305
Step 1250/5000 | Train (Unweighted): 5.160 | Probe: 5.161
Step 1500/5000 | Train (Unweighted): 5.053 | Probe: 5.109
Step 1750/5000 | Train (Unweighted): 4.848 | Probe: 5.052
Step 2000/5000 | Train (Unweighted): 4.926 | Probe: 4.996
Step 2250/5000 | Train (Unweighted): 4.909 | Probe: 4.940
Step 2500/5000 | Train (Unweighted): 4.808 | Probe: 4.927
Step 2750/5000 | Train (Unweighted): 4.847 | Probe: 4.885
Step 3000/5000 | Train (Unweighted): 4.689 | Probe: 4.816
Step 3250/5000 | Train (Unweighted): 4.836 | Probe: 4.796
Step 3500/5000 | Train (Unweighted): 4.780 | Probe: 4.788
Step 3750/5000 | Train (Unweighted): 4.525 | Probe: 4.755
Step 4000/5000 | Train (Unweighted): 4.630 | Probe: 4.745
Step 4250/5000 | Train (Unweighted): 4.616 | Probe: 4.736
Step 4500/5000 | Train (Unweighted): 4.536 | Probe: 4.724
Step 4750/5000 | Train (Unweighted): 4.684 | Probe: 4.722
Step 5000/5000 | Train (Unweighted): 4.524 | Probe: 4.723
```

### RMSprop (lr=1e-3)
```text
Step  250/5000 | Train: 7.503 | Probe: 7.619
Step  500/5000 | Train: 8.113 | Probe: 7.959
Step  750/5000 | Train: 7.276 | Probe: 7.373
Step 1000/5000 | Train: 7.020 | Probe: 7.139
Step 1250/5000 | Train: 7.010 | Probe: 7.089
Step 1500/5000 | Train: 6.852 | Probe: 6.860
Step 1750/5000 | Train: 6.814 | Probe: 6.750
Step 2000/5000 | Train: 6.631 | Probe: 6.671
Step 2250/5000 | Train: 6.660 | Probe: 6.672
Step 2500/5000 | Train: 6.673 | Probe: 6.653
Step 2750/5000 | Train: 6.574 | Probe: 6.653
Step 3000/5000 | Train: 6.595 | Probe: 6.602
Step 3250/5000 | Train: 6.660 | Probe: 6.612
Step 3500/5000 | Train: 6.523 | Probe: 6.603
Step 3750/5000 | Train: 6.601 | Probe: 6.592
Step 4000/5000 | Train: 6.566 | Probe: 6.589
Step 4250/5000 | Train: 6.462 | Probe: 6.575
Step 4500/5000 | Train: 6.681 | Probe: 6.581
Step 4750/5000 | Train: 6.557 | Probe: 6.582
Step 5000/5000 | Train: 6.661 | Probe: 6.581
```

### RMSprop (lr=1e-4)
```text
Step  250/5000 | Train: 7.486 | Probe: 7.536
Step  500/5000 | Train: 7.094 | Probe: 7.115
Step  750/5000 | Train: 6.984 | Probe: 6.984
Step 1000/5000 | Train: 6.930 | Probe: 6.930
Step 1250/5000 | Train: 7.030 | Probe: 6.972
Step 1500/5000 | Train: 7.043 | Probe: 6.961
Step 1750/5000 | Train: 7.162 | Probe: 7.012
Step 2000/5000 | Train: 7.030 | Probe: 6.974
Step 2250/5000 | Train: 7.026 | Probe: 6.979
Step 2500/5000 | Train: 7.130 | Probe: 6.993
Step 2750/5000 | Train: 7.018 | Probe: 6.998
Step 3000/5000 | Train: 7.094 | Probe: 6.991
Step 3250/5000 | Train: 7.189 | Probe: 7.030
Step 3500/5000 | Train: 7.083 | Probe: 7.038
Step 3750/5000 | Train: 7.139 | Probe: 7.062
Step 4000/5000 | Train: 7.144 | Probe: 7.078
Step 4250/5000 | Train: 7.043 | Probe: 7.081
Step 4500/5000 | Train: 7.244 | Probe: 7.091
Step 4750/5000 | Train: 7.157 | Probe: 7.093
Step 5000/5000 | Train: 7.261 | Probe: 7.095
```

### SGD Momentum (lr=0.01)
```text
Step  250/5000 | Train: 8.076 | Probe: 8.058
Step  500/5000 | Train: 7.615 | Probe: 7.626
Step  750/5000 | Train: 7.392 | Probe: 7.423
Step 1000/5000 | Train: 7.115 | Probe: 7.214
Step 1250/5000 | Train: 7.049 | Probe: 7.101
Step 1500/5000 | Train: 6.975 | Probe: 7.005
Step 1750/5000 | Train: 6.967 | Probe: 6.943
Step 2000/5000 | Train: 6.780 | Probe: 6.876
Step 2250/5000 | Train: 6.825 | Probe: 6.857
Step 2500/5000 | Train: 6.831 | Probe: 6.821
Step 2750/5000 | Train: 6.718 | Probe: 6.827
Step 3000/5000 | Train: 6.755 | Probe: 6.785
Step 3250/5000 | Train: 6.805 | Probe: 6.786
Step 3500/5000 | Train: 6.666 | Probe: 6.767
Step 3750/5000 | Train: 6.737 | Probe: 6.762
Step 4000/5000 | Train: 6.695 | Probe: 6.751
Step 4250/5000 | Train: 6.617 | Probe: 6.743
Step 4500/5000 | Train: 6.815 | Probe: 6.746
Step 4750/5000 | Train: 6.696 | Probe: 6.746
Step 5000/5000 | Train: 6.785 | Probe: 6.747
```

### SGD Momentum (lr=0.1)
```text
Step  250/5000 | Train: 7.309 | Probe: 7.426
Step  500/5000 | Train: 7.138 | Probe: 7.173
Step  750/5000 | Train: 7.072 | Probe: 7.101
Step 1000/5000 | Train: 6.996 | Probe: 7.096
Step 1250/5000 | Train: 7.064 | Probe: 7.088
Step 1500/5000 | Train: 7.040 | Probe: 7.018
Step 1750/5000 | Train: 6.962 | Probe: 7.059
Step 2000/5000 | Train: 6.843 | Probe: 6.924
Step 2250/5000 | Train: 6.851 | Probe: 6.888
Step 2500/5000 | Train: 6.861 | Probe: 6.886
Step 2750/5000 | Train: 6.756 | Probe: 6.870
Step 3000/5000 | Train: 6.739 | Probe: 6.782
Step 3250/5000 | Train: 6.781 | Probe: 6.769
Step 3500/5000 | Train: 6.609 | Probe: 6.695
Step 3750/5000 | Train: 6.652 | Probe: 6.666
Step 4000/5000 | Train: 6.611 | Probe: 6.662
Step 4250/5000 | Train: 6.516 | Probe: 6.639
Step 4500/5000 | Train: 6.698 | Probe: 6.635
Step 4750/5000 | Train: 6.571 | Probe: 6.620
Step 5000/5000 | Train: 6.662 | Probe: 6.620
```
