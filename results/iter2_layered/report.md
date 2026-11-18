# Harness report: iter2_layered

Real people: **414** (22777 trials, Findlater & Zhang 2020). All numbers are on held-out people (5 folds split by person). Bots click through the same targets as each human window.

## Gates

| gate | what | value | need | result |
|---|---|---|---|---|
| G1-fpr | false-positive rate on held-out real people, Fitts-only detector, 25 clicks | 0.014 | <= 0.03 | PASS |
| G2-fairness | worst false-positive rate over age bands, motor impairment and touchpad users, Fitts-only detector | 0.034 | <= 0.05 | PASS |
| G3-tier0 | detection of teleporting bots, Fitts-only detector, 25 clicks | 0.830 | >= 0.95 | OPEN (target not met) |
| G3-tier1 | detection of machine-timed movement bots, Fitts-only detector, 25 clicks | 0.841 | >= 0.95 | OPEN (target not met) |
| G3-tier2 | detection of 'human-like path' bots (bezier, wind), Fitts-only detector, 25 clicks | 0.506 | >= 0.95 | OPEN (target not met) |
| G4-tier3 | detection of adaptive bots (Fitts-aware, replay, simulator), Fitts+kinematics detector, 25 clicks | 0.694 | >= 0.5 | PASS |
| G5-age-band | age model: worst cross-validated error of age-band mean movement time (%) | 3.788 | <= 5.0 | PASS |
| G6-age-gain | age model: band-mean error relative to an age-blind model (ratio) | 0.227 | <= 0.5 | PASS |
| R1-fpr-full | regression floor: false-positive rate, Fitts+kinematics detector, 25 clicks | 0.014 | <= 0.03 | PASS |
| R2-tier1-full | regression floor: machine-timed bots, Fitts+kinematics detector, 25 clicks | 1.000 | >= 0.95 | PASS |
| R3-tier0-fitts50 | regression floor: teleporting bots, Fitts-only detector, 50 clicks | 0.971 | >= 0.9 | PASS |
| R4-tier2-fitts50 | regression floor: 'human-like path' bots, Fitts-only detector, 50 clicks | 0.796 | >= 0.7 | PASS |
| R5-tier3-full | regression floor: adaptive bots, Fitts+kinematics detector, 25 clicks | 0.694 | >= 0.6 | PASS |

## Detection rate by bot family (alarm threshold set for 1% false positives on training people)

### 10 clicks per decision

| bot family | tier | layered_fitts | layered_fitts_kin | generic_outlier |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 0.0% |
| teleport_jitter | 0 | 18.2% | 23.8% | 0.0% |
| const_speed | 1 | 100.0% | 100.0% | 0.0% |
| const_time | 1 | 100.0% | 100.0% | 0.0% |
| ease_tween | 1 | 16.5% | 100.0% | 0.0% |
| bezier | 2 | 12.1% | 9.5% | 0.0% |
| wind | 2 | 10.5% | 11.6% | 0.0% |
| fitts_aware | 3 | 0.6% | 99.9% | 0.0% |
| replay | 3 | 9.9% | 7.6% | 0.0% |
| sim_bot | 3 | 0.8% | 38.5% | 0.0% |
| **false positives (real people)** | | **1.4%** | **1.4%** | **0.0%** |

### 25 clicks per decision

| bot family | tier | layered_fitts | layered_fitts_kin | generic_outlier |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 20.0% |
| teleport_jitter | 0 | 66.1% | 66.4% | 20.0% |
| const_speed | 1 | 100.0% | 100.0% | 20.0% |
| const_time | 1 | 100.0% | 100.0% | 20.0% |
| ease_tween | 1 | 52.3% | 100.0% | 0.0% |
| bezier | 2 | 33.5% | 26.0% | 0.0% |
| wind | 2 | 67.8% | 76.0% | 0.0% |
| fitts_aware | 3 | 0.0% | 100.0% | 0.0% |
| replay | 3 | 19.8% | 16.4% | 0.0% |
| sim_bot | 3 | 0.0% | 91.8% | 0.0% |
| **false positives (real people)** | | **1.4%** | **1.4%** | **0.7%** |

### 50 clicks per decision

| bot family | tier | layered_fitts | layered_fitts_kin | generic_outlier |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 59.9% |
| teleport_jitter | 0 | 94.2% | 92.3% | 59.9% |
| const_speed | 1 | 100.0% | 100.0% | 59.9% |
| const_time | 1 | 100.0% | 100.0% | 59.9% |
| ease_tween | 1 | 83.8% | 100.0% | 0.0% |
| bezier | 2 | 62.1% | 51.7% | 0.0% |
| wind | 2 | 97.1% | 93.0% | 0.0% |
| fitts_aware | 3 | 0.2% | 100.0% | 0.0% |
| replay | 3 | 26.6% | 22.0% | 0.0% |
| sim_bot | 3 | 0.0% | 96.4% | 0.0% |
| **false positives (real people)** | | **1.9%** | **2.9%** | **1.2%** |

## Does conditioning the Fitts tests on age help?

Time-scale statistics are divided by the expected slowing at the account's age before testing. A bot inherits the age of the account it runs on.

| clicks | variant | FPR all | worst age-band FPR | tier0 | tier1 | tier2 | tier3 |
|---|---|---|---|---|---|---|---|
| 10 | age_blind | 1.4% | 1.8% | 59.1% | 72.2% | 11.3% | 3.8% |
| 10 | age_conditioned | 1.3% | 1.6% | 58.7% | 72.1% | 11.0% | 3.5% |
| 25 | age_blind | 1.4% | 3.4% | 83.0% | 84.1% | 50.6% | 6.6% |
| 25 | age_conditioned | 1.4% | 4.0% | 80.7% | 82.8% | 47.5% | 6.2% |
| 50 | age_blind | 1.9% | 4.5% | 97.1% | 94.6% | 79.6% | 8.9% |
| 50 | age_conditioned | 2.2% | 4.5% | 97.0% | 92.6% | 78.9% | 9.2% |

## Who gets falsely flagged (25 clicks)

### layered_fitts

| group | windows | false-positive rate |
|---|---|---|
| all | 828 | 1.4% |
| age_18_30 | 228 | 0.4% |
| age_30_45 | 206 | 1.9% |
| age_45_60 | 176 | 3.4% |
| age_60_70 | 158 | 0.0% |
| age_70_90 | 60 | 1.7% |
| motor_impairment | 394 | 1.5% |
| no_motor_impairment | 434 | 1.4% |
| mouse | 642 | 1.7% |
| touchpad | 164 | 0.6% |

### layered_fitts_kin

| group | windows | false-positive rate |
|---|---|---|
| all | 828 | 1.4% |
| age_18_30 | 228 | 1.3% |
| age_30_45 | 206 | 2.4% |
| age_45_60 | 176 | 2.3% |
| age_60_70 | 158 | 0.0% |
| age_70_90 | 60 | 0.0% |
| motor_impairment | 394 | 0.8% |
| no_motor_impairment | 434 | 2.1% |
| mouse | 642 | 1.6% |
| touchpad | 164 | 1.2% |

## Supervised reference (gradient boosting, 25 clicks)

Trained on humans plus bots. *Leave-family-out*: the family in the row was removed from training, so the number is performance on an attack never seen before.

### features: fitts  (FPR 2.1%)

| bot family | tier | seen in training | never seen (leave-family-out) | FPR in that run |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 2.1% |
| teleport_jitter | 0 | 98.8% | 87.8% | 2.2% |
| const_speed | 1 | 98.9% | 83.6% | 2.2% |
| const_time | 1 | 100.0% | 86.8% | 1.8% |
| ease_tween | 1 | 93.5% | 76.4% | 2.1% |
| bezier | 2 | 89.6% | 32.7% | 1.6% |
| wind | 2 | 93.4% | 41.3% | 2.3% |
| fitts_aware | 3 | 50.1% | 35.4% | 2.4% |
| replay | 3 | 90.9% | 37.1% | 2.1% |
| sim_bot | 3 | 25.6% | 14.6% | 1.6% |

### features: fitts_kin  (FPR 2.2%)

| bot family | tier | seen in training | never seen (leave-family-out) | FPR in that run |
|---|---|---|---|---|
| teleport | 0 | 100.0% | 100.0% | 2.4% |
| teleport_jitter | 0 | 100.0% | 94.6% | 1.8% |
| const_speed | 1 | 100.0% | 99.8% | 2.2% |
| const_time | 1 | 100.0% | 92.9% | 2.1% |
| ease_tween | 1 | 100.0% | 100.0% | 2.1% |
| bezier | 2 | 99.9% | 60.3% | 1.9% |
| wind | 2 | 100.0% | 92.9% | 2.1% |
| fitts_aware | 3 | 100.0% | 100.0% | 1.8% |
| replay | 3 | 99.3% | 0.0% | 2.2% |
| sim_bot | 3 | 99.8% | 63.6% | 2.3% |

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

| age | a (ms) | b (ms/bit) | MT (ms) | 95% of people | throughput (bits/s) | hold (ms) | max clicks/s |
|---|---|---|---|---|---|---|---|
| 5 | 1068 | 472 | 2841 | 929-8688 | 1.32 | 87 | |
| 8 | 490 | 216 | 1303 | 426-3983 | 2.88 | 87 | |
| 12 | 345 | 153 | 919 | 300-2809 | 4.09 | 87 | |
| 16 | 313 | 139 | 834 | 273-2550 | 4.50 | 87 | |
| 20 | 305 | 135 | 812 | 493-1336 | 4.63 | 87 | |
| 30 | 302 | 134 | 803 | 488-1322 | 4.67 | 87 | 4.9 |
| 40 | 324 | 143 | 863 | 524-1420 | 4.35 | 96 | 4.7 |
| 50 | 349 | 154 | 927 | 563-1526 | 4.05 | 106 | 4.4 |
| 60 | 375 | 166 | 996 | 605-1640 | 3.77 | 116 | 4.1 |
| 70 | 402 | 178 | 1071 | 650-1762 | 3.51 | 128 | 3.9 |
| 80 | 432 | 191 | 1150 | 699-1893 | 3.26 | 140 | |

Literature-based parts (within-study ratios, leave-one-study-out):

- mt: 5 studies / 12 groups, ages 4-78, leave-one-study-out error 51% (log); ageing +1.53%/year after 30; factor vs age 25: age 5: x3.53, age 8: x1.62, age 12: x1.14, age 25: x1.00, age 50: x1.36, age 70: x1.84
- tap: 2 studies / 7 groups, ages 24-80, leave-one-study-out error 9% (log); ageing -0.59%/year after 30; factor vs age 25: age 25: x1.00, age 50: x0.89, age 70: x0.79
