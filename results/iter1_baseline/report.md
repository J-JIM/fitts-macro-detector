# Harness report: iter1_baseline

Real people: **414** (22777 trials, Findlater & Zhang 2020). All numbers are on held-out people (5 folds split by person). Bots click through the same targets as each human window.

## Gates

| gate | what | value | need | result |
|---|---|---|---|---|
| G1-fpr | false-positive rate on held-out real people, Fitts-only envelope, 25 clicks | 0.011 | <= 0.03 | PASS |
| G2-fairness | worst false-positive rate over age bands and motor-impairment groups, Fitts-only envelope | 0.023 | <= 0.05 | PASS |
| G3-tier0 | detection of teleporting bots, Fitts-only envelope, 25 clicks | 0.501 | >= 0.95 | FAIL |
| G3-tier1 | detection of machine-timed movement bots, Fitts-only envelope, 25 clicks | 0.519 | >= 0.95 | FAIL |
| G3-tier2 | detection of 'human-like path' bots (bezier, wind), Fitts-only envelope, 25 clicks | 0.000 | >= 0.95 | FAIL |
| G4-tier3 | detection of adaptive bots (Fitts-aware, replay, simulator), Fitts+kinematics envelope, 25 clicks | 0.000 | >= 0.5 | miss (stretch) |
| G5-age-band | age model: worst cross-validated error of age-band mean movement time (%) | 3.788 | <= 5.0 | PASS |
| G6-age-gain | age model: band-mean error relative to an age-blind model (ratio) | 0.227 | <= 0.5 | PASS |

## Detection rate by bot family (alarm threshold set for 1% false positives on training people)

### 10 clicks per decision

| bot family | tier | envelope_fitts | envelope_fitts_kin | mahalanobis_fitts | mahalanobis_fitts_kin |
|---|---|---|---|---|---|
| teleport | 0 | 100.0% | 0.0% | 100.0% | 100.0% |
| teleport_jitter | 0 | 0.2% | 0.0% | 2.1% | 98.2% |
| const_speed | 1 | 11.9% | 0.0% | 100.0% | 100.0% |
| const_time | 1 | 100.0% | 0.0% | 100.0% | 100.0% |
| ease_tween | 1 | 0.2% | 0.0% | 0.2% | 0.2% |
| bezier | 2 | 0.2% | 0.0% | 0.2% | 0.2% |
| wind | 2 | 0.2% | 0.0% | 0.2% | 0.2% |
| fitts_aware | 3 | 0.2% | 0.0% | 0.2% | 0.1% |
| replay | 3 | 0.5% | 0.0% | 0.6% | 0.2% |
| sim_bot | 3 | 0.2% | 0.0% | 0.2% | 0.1% |
| **false positives (real people)** | | **0.8%** | **0.0%** | **1.2%** | **1.4%** |

### 25 clicks per decision

| bot family | tier | envelope_fitts | envelope_fitts_kin | mahalanobis_fitts | mahalanobis_fitts_kin |
|---|---|---|---|---|---|
| teleport | 0 | 100.0% | 20.0% | 100.0% | 100.0% |
| teleport_jitter | 0 | 0.1% | 20.0% | 1.4% | 100.0% |
| const_speed | 1 | 55.7% | 20.0% | 100.0% | 100.0% |
| const_time | 1 | 100.0% | 20.0% | 100.0% | 100.0% |
| ease_tween | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| bezier | 2 | 0.0% | 0.0% | 0.0% | 0.0% |
| wind | 2 | 0.0% | 0.0% | 0.0% | 0.0% |
| fitts_aware | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| replay | 3 | 0.4% | 0.0% | 0.6% | 0.1% |
| sim_bot | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| **false positives (real people)** | | **1.1%** | **0.7%** | **1.2%** | **1.3%** |

### 50 clicks per decision

| bot family | tier | envelope_fitts | envelope_fitts_kin | mahalanobis_fitts | mahalanobis_fitts_kin |
|---|---|---|---|---|---|
| teleport | 0 | 100.0% | 59.9% | 100.0% | 100.0% |
| teleport_jitter | 0 | 8.0% | 59.9% | 15.9% | 100.0% |
| const_speed | 1 | 100.0% | 59.9% | 100.0% | 100.0% |
| const_time | 1 | 100.0% | 59.9% | 100.0% | 100.0% |
| ease_tween | 1 | 0.0% | 0.0% | 0.0% | 43.5% |
| bezier | 2 | 0.0% | 0.0% | 0.0% | 1.7% |
| wind | 2 | 0.0% | 0.0% | 0.0% | 0.0% |
| fitts_aware | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| replay | 3 | 1.4% | 0.2% | 1.0% | 0.7% |
| sim_bot | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| **false positives (real people)** | | **1.4%** | **1.4%** | **1.4%** | **1.4%** |

## Who gets falsely flagged (25 clicks)

### envelope_fitts

| group | windows | false-positive rate |
|---|---|---|
| all | 828 | 1.1% |
| age_18_30 | 228 | 0.4% |
| age_30_45 | 206 | 1.9% |
| age_45_60 | 176 | 1.7% |
| age_60_70 | 158 | 0.6% |
| age_70_90 | 60 | 0.0% |
| motor_impairment | 394 | 2.3% |
| no_motor_impairment | 434 | 0.0% |
| mouse | 642 | 1.1% |
| touchpad | 164 | 0.0% |

### envelope_fitts_kin

| group | windows | false-positive rate |
|---|---|---|
| all | 828 | 0.7% |
| age_18_30 | 228 | 0.0% |
| age_30_45 | 206 | 1.0% |
| age_45_60 | 176 | 2.3% |
| age_60_70 | 158 | 0.0% |
| age_70_90 | 60 | 0.0% |
| motor_impairment | 394 | 1.3% |
| no_motor_impairment | 434 | 0.2% |
| mouse | 642 | 0.6% |
| touchpad | 164 | 1.2% |

## Supervised reference (gradient boosting, 25 clicks)

Trained on humans plus bots. *Leave-family-out*: the family in the row was removed from training, so the number is performance on an attack never seen before.

### features: fitts  (FPR 1.4%)

| bot family | tier | seen in training | never seen (leave-family-out) | FPR in that run |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 1.4% |
| teleport_jitter | 0 | 100.0% | 95.5% | 1.3% |
| const_speed | 1 | 99.9% | 99.4% | 1.2% |
| const_time | 1 | 100.0% | 99.9% | 1.3% |
| ease_tween | 1 | 96.3% | 83.3% | 1.3% |
| bezier | 2 | 93.0% | 35.1% | 1.6% |
| wind | 2 | 95.8% | 57.7% | 1.7% |
| fitts_aware | 3 | 71.0% | 30.9% | 1.3% |
| replay | 3 | 91.1% | 17.8% | 1.6% |
| sim_bot | 3 | 35.4% | 24.9% | 1.9% |

### features: fitts_kin  (FPR 1.4%)

| bot family | tier | seen in training | never seen (leave-family-out) | FPR in that run |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 1.7% |
| teleport_jitter | 0 | 100.0% | 99.4% | 1.4% |
| const_speed | 1 | 100.0% | 100.0% | 1.7% |
| const_time | 1 | 100.0% | 95.4% | 1.6% |
| ease_tween | 1 | 100.0% | 100.0% | 1.4% |
| bezier | 2 | 99.8% | 20.3% | 2.4% |
| wind | 2 | 100.0% | 97.9% | 1.8% |
| fitts_aware | 3 | 100.0% | 100.0% | 1.3% |
| replay | 3 | 99.4% | 0.0% | 1.9% |
| sim_bot | 3 | 99.8% | 47.8% | 1.9% |

## Age model

Fitted on 414 people aged 18-83; cross-validated by person. Chosen variant: **both_shared** (the data cannot separate ageing of the intercept from ageing of the slope: the task only spans 1.9-4.1 bits, so proportional slowing is assumed).

| variant | person-level median error | age-band mean error | worst age band |
|---|---|---|---|
| age_blind | 16.9% | 7.7% | 13.7% |
| intercept_only | 14.7% | 1.8% | 3.2% |
| slope_only | 14.7% | 1.8% | 3.4% |
| both_separate | 14.5% | 1.8% | 3.5% |
| both_shared | 14.6% | 1.7% | 3.8% |

Held-out age-band means (chosen variant):

| age band | people | observed MT | predicted MT | error |
|---|---|---|---|---|
| 18-30 | 114 | 760 ms | 731 ms | -3.8% |
| 30-40 | 71 | 793 ms | 796 ms | +0.4% |
| 40-50 | 56 | 864 ms | 873 ms | +1.0% |
| 50-60 | 64 | 969 ms | 945 ms | -2.5% |
| 60-70 | 79 | 1053 ms | 1029 ms | -2.2% |
| 70-90 | 30 | 1058 ms | 1064 ms | +0.5% |

- reference (age <= 30, mouse, no motor impairment): MT = 303 ms + 134 ms/bit x ID
- ageing after 30: intercept +7.4% and slope +7.4% per decade
- touchpad x1.24, self-reported motor impairment x1.19
- button hold time: 87 ms at <= 30, +9.9% per decade
- spread between people of the same age: s.d. of log MT = 0.25

Predicted values by age (D = 400 px, W = 32 px, ID = 3.75 bits):

| age | a (ms) | b (ms/bit) | MT (ms) | 95% of people | throughput (bits/s) | hold (ms) |
|---|---|---|---|---|---|---|
| 20 | 303 | 134 | 805 | 489-1325 | 4.67 | 87 |
| 30 | 303 | 134 | 805 | 489-1325 | 4.67 | 87 |
| 40 | 325 | 144 | 865 | 525-1423 | 4.34 | 96 |
| 50 | 349 | 154 | 929 | 565-1529 | 4.04 | 106 |
| 60 | 375 | 166 | 998 | 607-1643 | 3.76 | 116 |
| 70 | 403 | 178 | 1073 | 652-1766 | 3.50 | 128 |
| 80 | 433 | 192 | 1153 | 700-1897 | 3.26 | 140 |
