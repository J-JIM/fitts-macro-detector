# fittsguard — Fitts의 법칙으로 매크로 적발하기

사람이 마우스로 목표를 클릭할 때 걸리는 시간은 **멀수록, 작을수록** 길어진다.

    MT = a + b · log2(D / W + 1)        (Fitts의 법칙, D = 거리, W = 목표 폭)

매크로는 이 법칙을 지킬 이유가 없다. 좌표를 찍거나, 일정한 속도로 움직이거나, 지연 시간을 난수로 뽑는다.
`fittsguard`는 클릭 기록이 이 법칙과 맞는지를 검정해 매크로를 가려내고,
나이에 따라 사람의 이동 시간·클릭 속도가 어떻게 달라지는지 예측하는 모델을 함께 제공한다.

- 탐지기는 **사람 데이터만으로** 기준을 세운다. 봇을 학습에 쓰지 않으므로 모든 적발률은 "처음 보는 공격"에 대한 수치다.
- 사람 기준선과 나이 모델은 **실제 사람 414명**(18~83세, 22,777회 클릭)의 공개 데이터로 만들고 검증했다.
- 모든 수치는 `python harness/run.py` 한 번으로 재현된다. 명세는 [HARNESS.md](HARNESS.md), 반복 기록은 [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

## 결과 요약

학습에 쓰지 않은 사람으로 측정(사람 단위 5겹 교차검증). 봇은 각 사람과 **같은 목표 순서**를 클릭한다.
경보 임계값은 학습 쪽 사람의 오탐 1%에 맞췄고, 표의 오탐률은 평가 쪽 사람에게서 실제로 나온 값이다.

| 봇 계열 | 무엇을 하는가 | Fitts만 · 25클릭 | Fitts만 · 50클릭 | Fitts+운동학 · 25클릭 | Fitts+운동학 · 50클릭 |
|---|---|---|---|---|---|
| `teleport` | 좌표 지정 후 즉시 클릭 | 100% | 100% | 100% | 100% |
| `teleport_jitter` | 순간이동 + 무작위 지연·위치 | 66% | 94% | 66% | 92% |
| `const_speed` | 등속 직선 이동 | 100% | 100% | 100% | 100% |
| `const_time` | 고정 시간 이동 (`moveTo(duration=T)`) | 100% | 100% | 100% | 100% |
| `ease_tween` | 가감속 트윈 + 무작위 시간 | 52% | 84% | 100% | 100% |
| `bezier` | 휘는 경로 + 무작위 시간 | 34% | 62% | 26% | 52% |
| `wind` | 중력·바람 물리 (WindMouse 계열) | 68% | 97% | 76% | 93% |
| `fitts_aware` | Fitts 선대로 시간을 정함 (적응형) | 0% | 0% | 100% | 100% |
| `replay` | 사람 궤적을 녹화해 재생 (적응형) | 20% | 27% | 16% | 22% |
| `sim_bot` | 하위운동 시뮬레이터 (적응형) | 0% | 0% | 92% | 96% |
| **실제 사람 오탐률** | | **1.4%** | **1.9%** | **1.4%** | **2.9%** |

읽는 법

- **기계적 타이밍(티어 0~1)은 Fitts 검정만으로 거의 다 잡힌다.** 궤적이 없어도, 클릭 시각·위치와 목표 크기만 있으면 된다.
- **지연을 난수로 뽑는 봇은 클릭이 쌓여야 잡힌다.** 25클릭 34~68%, 50클릭 62~97%. 사람의 Fitts 기울기가 0이 아니라는 것을 통계적으로 확인하는 데 표본이 필요하기 때문이다.
- **Fitts 법칙을 흉내 내는 봇은 Fitts 검정으로 못 잡는다(0%).** 원리상 당연하다. 이들은 운동학 검정(너무 매끄러움·너무 곧음·판에 박힘)이 잡는다.
- **사람 궤적 재생은 가장 어렵다(16~27%).** 운동학은 진짜 사람 것이라 통과하고, 재생된 시간이 지금 목표와 안 맞는다는 Fitts 불일치로만 일부 잡힌다.
- 비교: 같은 특징에 일반적인 이상치 탐지기를 쓰면 25클릭에서 0~20%. 봇을 모아 학습한 지도 모델은 본 적 있는 봇을 99~100% 잡지만, 그 계열을 학습에서 빼면 **replay 0%, bezier 60%, sim_bot 64%**로 떨어진다.

### 누가 오탐을 당하는가 (25클릭, Fitts+운동학)

| 집단 | 창 수 | 오탐률 |
|---|---|---|
| 전체 | 828 | 1.4% |
| 18~29세 / 30~44세 / 45~59세 / 60~69세 / 70세 이상 | 228 / 206 / 176 / 158 / 60 | 1.3% / 2.4% / 2.3% / 0.0% / 0.0% |
| 운동 장애 자기보고 있음 / 없음 | 394 / 434 | 0.8% / 2.1% |
| 마우스 / 터치패드 | 642 / 164 | 1.6% / 1.2% |

고령자와 운동 장애가 있는 사람이 더 걸리지 않는다. 검정이 **기계 쪽으로 벗어난 것만** 세기 때문이다 —
느리고, 멈칫거리고, 빗나가는 것은 점수를 올리지 않는다.

### 클릭이 몇 번 필요한가 (모델 기반 외삽)

실측 데이터는 난이도 1.9~4.1비트, 사람당 55회가 한계다. 각 실제 사람의 Fitts 선과 잔차를 그대로 써서 다른 과제에 다시 태운 추정치다.
같은 방법으로 원래 과제를 재현하면 실측과 비슷하거나 조금 낮게 나온다(25클릭: 실측 34% / 52% → 추정 33% / 53%, 50클릭: 실측 62% / 84% → 추정 52% / 78%).
1% 오탐 꼬리를 소수의 사람이 정하므로 값이 흔들린다 — 시드 3개의 평균이고 괄호는 최소~최대다. **경향을 보는 표**다.

`task_blind_timing` 검정 하나의 적발률, 오탐 1%:

| 과제의 난이도 범위 | 봇의 타이밍 | 10클릭 | 25클릭 | 50클릭 | 100클릭 | 200클릭 |
|---|---|---|---|---|---|---|
| 이 데이터의 과제 (1.9~4.1비트) | 트윈형 (`ease_tween`) | 20% (16~26) | 53% (45~58) | 78% (75~81) | 91% (88~95) | 95% (87~99) |
| 일반적인 웹 UI (1.0~5.2비트) | 〃 | 38% (35~40) | 62% (53~78) | 90% (85~96) | 95% (87~99) | 89% (80~94) |
| 작은 목표를 섞은 UI (0.7~6.2비트) | 〃 | 48% (43~52) | 80% (77~85) | 90% (82~93) | 98% (97~99) | 96% (93~98) |
| 이 데이터의 과제 (1.9~4.1비트) | 베지어형 (`bezier`) | 14% (12~18) | 33% (28~38) | 52% (49~54) | 68% (64~74) | 82% (69~90) |
| 일반적인 웹 UI (1.0~5.2비트) | 〃 | 26% (25~27) | 40% (33~54) | 70% (62~78) | 81% (71~89) | 76% (63~83) |
| 작은 목표를 섞은 UI (0.7~6.2비트) | 〃 | 32% (28~36) | 57% (52~61) | 71% (62~78) | 90% (89~92) | 85% (81~88) |

- **난이도 범위를 넓히면 필요한 클릭이 절반쯤으로 준다.** 작은 목표를 섞은 UI의 25클릭(80% / 57%)이 원래 과제의 50클릭(78% / 52%)과 비슷하다.
- 100클릭 이후로는 더 오르지 않는다. 사람 중 일부(1~2%)는 이 범위에서 기울기가 실제로 0에 가깝게 추정되고, 이들이 임계값을 정하기 때문이다. Fitts 타이밍 검정 하나로 100%에 가는 길은 없다.

## 어떻게 판정하는가

이름 붙은 **단측 검정 8개**. 하나라도 선을 넘으면 경보이고, 어느 검정이 어떤 수치로 걸렸는지 그대로 보여 준다.

| 계층 | 검정 | 묻는 것 | 필요한 입력 |
|---|---|---|---|
| Fitts | `superhuman_throughput` | 사람이 낼 수 없는 초당 비트 수인가 | 클릭 시각·위치, 목표 중심·폭 |
| Fitts | `task_blind_timing` | 시간이 난이도에 따라 늘고 목표 폭에 따라 줄어드는가 (기울기의 신뢰한계) | 〃 |
| Fitts | `machine_regular_timing` | Fitts 선 주변의 흩어짐이 사람보다 작은가 | 〃 |
| Fitts | `machine_precise_clicks` | 클릭 지점의 흩어짐이 목표 크기에 비해 너무 작은가 | 〃 |
| 운동학 | `too_smooth` | 속도 곡선이 손이 내는 것보다 깔끔한 종 모양인가 | + 커서 궤적 |
| 운동학 | `too_straight` | 경로가 손이 내는 것보다 곧은가 | 〃 |
| 운동학 | `stereotyped` | 모든 움직임이 서로 판박이인가 | 〃 |
| 운동학 | `no_verification_pause` | 도착하자마자 버튼이 눌리는가 | 〃 |

기준은 사람 데이터에서만 나온다: 사람 범위와 기계 범위가 겹치지 않는 검정은 사람 범위 바깥에 선을 긋고,
겹치는 검정만 오탐 예산(기본 1%)을 나눠 쓴다.

## 나이 모델

414명을 사람 단위로 교차검증한 결과, **연령대 평균 이동 시간**의 예측 오차:

| | 연령대 평균 오차 | 최악 연령대 | 개인 단위 오차(중앙값) |
|---|---|---|---|
| 나이를 모르는 모델 | 7.7% | 13.7% | 16.9% |
| 나이 모델 | **1.7%** | **3.8%** | 14.6% |

연령대 평균은 잘 맞히지만 **개인**은 그렇지 않다. 같은 나이 안의 개인차(로그 표준편차 0.25)가 나이 효과보다 크다.

예측값 (D = 400 px, W = 32 px, 난이도 3.75비트, 마우스):

| 나이 | Fitts 선 (ms) | 이동 시간 | 사람의 95% | 처리량 | 버튼 누름 | 최대 연타 |
|---|---|---|---|---|---|---|
| 5 ⚠ | 1068 + 472·ID | 2841 ms | 929~8688 | 1.3 bits/s | — | — |
| 8 ⚠ | 490 + 216·ID | 1303 ms | 426~3983 | 2.9 | — | — |
| 12 ⚠ | 345 + 153·ID | 919 ms | 300~2809 | 4.1 | — | — |
| 20 | 305 + 135·ID | 812 ms | 493~1336 | 4.6 | 87 ms | 4.9 /s |
| 30 | 302 + 134·ID | 803 ms | 488~1322 | 4.7 | 87 ms | 4.9 /s |
| 40 | 324 + 143·ID | 863 ms | 524~1420 | 4.4 | 96 ms | 4.7 /s |
| 50 | 349 + 154·ID | 927 ms | 563~1526 | 4.1 | 106 ms | 4.4 /s |
| 60 | 375 + 166·ID | 996 ms | 605~1640 | 3.8 | 116 ms | 4.1 /s |
| 70 | 402 + 178·ID | 1071 ms | 650~1762 | 3.5 | 128 ms | 3.9 /s |
| 80 | 432 + 191·ID | 1150 ms | 699~1893 | 3.3 | 140 ms | — |

- **18~83세(이동 시간·버튼 누름)**: 실측 414명. 30세 이후 10년마다 이동 시간 +7.4%, 버튼 누름 시간 +9.9%. 터치패드 ×1.24, 운동 장애 자기보고 ×1.19.
- **⚠ 18세 미만**: 실측이 없어 발표된 연구 2건의 아동 3집단(4·5·13세)의 성인 대비 비율로 이었다. 참고용이다.
- **최대 연타(한 자리에서 최대한 빨리)**: 마우스 실측 연구를 찾지 못해 손가락 태핑 검사 규준을 상한의 근사로 썼다(30세 이후 연 −0.6%).
- **알 수 없었던 것**: 나이가 Fitts 선의 절편을 늘리는지 기울기를 늘리는지. 이 데이터의 난이도 범위(1.9~4.1비트)에서는 두 모델의 오차가 같아 전체가 비례해 느려진다고 가정했다.
- **문헌과 다른 점**: 1997~2013년 실험실 연구들은 70대가 20대보다 2.0~2.5배 느리다고 보고하지만, 2015~16년에 자기 집에서 자기 마우스로 한 이 데이터에서는 1.4배다. 문헌의 연령 곡선은 연구끼리도 맞지 않아(연구 하나를 빼고 맞춘 오차 51%) 성인 구간에는 쓰지 않았다.

## 한계

- **목표의 위치와 크기를 알아야 한다.** 버튼·아이콘의 좌표를 아는 서비스(웹 페이지, 게임 UI)에서 쓰는 방법이다. 자유로운 마우스 움직임만으로는 Fitts 검정을 할 수 없다.
- **최소 25클릭.** 10클릭에서는 지연을 난수로 뽑는 봇(teleport_jitter·bezier·wind·replay)을 8~24%밖에 못 잡는다.
- **봇 동물원은 직접 구현한 것이다.** 공개 도구(pyautogui, WindMouse, ghost-cursor 등)의 원리를 따랐지만 그 도구가 낸 궤적 자체는 아니다. 검정 구성은 이 동물원을 보며 골랐으므로, 동물원 밖 봇에 대한 수치는 이보다 낮을 수 있다.
- **사람 데이터는 한 종류의 과제다.** 원형 목표 5가지 조합(거리 250/500 px × 폭 32/64/96 px). 다른 UI에서는 사람 기준선을 다시 맞춰야 한다 — 수집 페이지가 그 용도다.
- **적응형 공격자에게는 한계가 분명하다.** 사람 궤적 재생은 16~27%, 난이도가 맞는 녹화본을 골라 재생하면 그 이하다. 단독 방어선이 아니라 여러 신호 중 하나로 써야 한다.
- 오탐 1%는 창 하나 기준이다. 실제 서비스에서는 여러 창의 결과를 묶고 사람의 확인 절차를 두어야 한다.

## 설치와 사용

```bash
git clone https://github.com/J-JIM/fitts-macro-detector.git
cd fitts-macro-detector
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

### 판정

```bash
# 데모용 세션 만들기 (사람 시뮬레이션 또는 봇 계열)
.venv/bin/fittsguard simulate --agent ease_tween -n 50 -o bot.json
.venv/bin/fittsguard score bot.json          # 걸리면 종료 코드 1
```

```
BOT  (50 clicks; strongest test at 1.30x its alarm line)
  [FIRED] stereotyped               1.30x  every movement looks like every other movement
  [FIRED] too_smooth                1.28x  speed profiles are cleaner bells than a hand produces
  [FIRED] task_blind_timing         1.19x  time does not depend on distance and target size the way Fitts' law requires
           evidence: b_ucb=-0.00648, bw_lcb=-0.0131
```

```python
from fittsguard import AgeMotorModel, Session, load_detector

verdict = load_detector().judge(Session.load("session.json"))
print(verdict.is_bot, verdict.fired)          # True ['stereotyped', 'too_smooth', ...]

model = AgeMotorModel.load()
print(model.movement_time(age=68, distance=600, width=20)["mt"])   # 1.266 (seconds)
```

세션 JSON은 시행의 목록이다. 시행 하나 = 이전 클릭부터 이번 클릭까지.

```json
{"trials": [{"t": [0, 0.008, ...], "x": [...], "y": [...],
             "target": {"x": 640, "y": 300, "w": 48},
             "click": {"t_down": 0.83, "t_up": 0.92, "x": 644, "y": 297}}]}
```

### 나이별 예측

```bash
.venv/bin/fittsguard predict --age 68 -d 600 -w 20
```

```
age 68, D=600px, W=20px  (ID = 4.95 bits)
  Fitts line        MT = 397 ms + 175 ms/bit x ID
  movement time     1266 ms   (95% of people: 769 to 2083 ms)
  pointing speed    0.79 clicks/s on this target,  throughput 3.91 bits/s
  button hold time  125 ms   (95% of people: 64 to 245 ms)
  max click rate    4.0 clicks/s on one spot   (95% of people: 2.9 to 5.3)
```

### 하네스 다시 돌리기

```bash
.venv/bin/python harness/fetch_data.py                 # 공개 데이터 약 250 MB 내려받기
.venv/bin/python harness/run.py --name my_run          # results/my_run/report.md
.venv/bin/python harness/power_analysis.py
```

### 내 데이터 모으기

`collector/index.html`을 브라우저로 열면 같은 형식의 포인팅 과제를 직접 하고 JSON으로 저장할 수 있다(나이 입력 선택).
`fittsguard score`에 바로 넣을 수 있다.

## 저장소 구조

```
src/fittsguard/     schema · fitts · features · detector · age_model · human_sim · bots · simulate · cli
harness/            run.py(진입점) · evaluate.py · datasets.py · age_experiment.py · power_analysis.py · criteria.json
data/literature/    나이별 문헌 수치표 (출처·확인 위치 포함)
collector/          브라우저 수집 페이지
results/            하네스 실행 결과 (report.md, metrics.json)
docs/RESEARCH_LOG.md
```

## 데이터와 인용

- 사람 데이터: Findlater, L. and Zhang, L. (2020). *Input Accessibility: A Large Dataset and Summary Analysis of Age, Motor Ability and Input Performance.* ACM ASSETS 2020. CC BY 4.0. <https://github.com/leahkf/mouseandtouchinput> — 이 저장소에는 원본을 넣지 않았고, 포함된 `human_reference.json`·`age_model.json`은 그 데이터에서 계산한 분위수와 회귀 계수다.
- 나이별 문헌 수치의 출처는 `data/literature/age_groups.csv`의 각 행에 있다.
- Fitts, P. M. (1954). *The information capacity of the human motor system in controlling the amplitude of movement.* J. Exp. Psychol. 47(6).
- MacKenzie, I. S. (1992). *Fitts' law as a research and design tool in human-computer interaction.* Human-Computer Interaction 7.

## 라이선스

MIT. [LICENSE](LICENSE) 참조.
