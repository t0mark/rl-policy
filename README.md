# PGTT

**Procedural Ground and Terrain Toolbox** — WFC(Wave Function Collapse) 알고리즘으로 지형을 절차적으로 생성하고, 생성된 지형 위에서 사족 보행 로봇(Go2-inspired)을 MuJoCo로 시뮬레이션하는 툴킷입니다.

- **WFC 지형 생성**: 타일 간 인접 제약을 전파해 계단, 평지, 경사 등 다양한 지형을 자동 배치
- **Rough Ground 생성**: N×N 박스를 무작위 크기·위치·회전으로 배치해 울퉁불퉁한 지면 생성
- **사족 보행 시뮬레이션**: 트롯 보행 CPG + PD 제어로 로봇이 생성된 지형을 주행

## 로컬 실행

```bash
pip install -r requirements.txt
python benchmark/run_benchmark.py
```

## Docker 실행

```bash
docker compose -f docker/docker-compose.yml up --build
```

결과는 `results/benchmark_results.csv` 에 저장됩니다.

## 벤치마크 측정 항목

- **WFC** : 그리드 크기 5×5 ~ 15×15, 각 7회 반복, 평균 + 표준편차
- **AddRoughGround** : 박스 수 5×5 ~ 30×30, 각 7회 반복, 평균 + 표준편차
- 메모리: `tracemalloc` peak KB
- 출력: `results/benchmark_results.csv`

## 로봇 주행 시뮬레이션

최적화된 `AddRoughGround` 지형 위를 Go2-inspired 사족 보행 로봇이 트롯 보행으로 주행하는 MuJoCo 시뮬레이션입니다.

![로봇 주행](results/robot_walk.gif)

```bash
# Docker로 시뮬레이션 실행 (MP4 + GIF 생성)
docker compose -f docker/docker-compose.yml run simulate
```

결과물: `results/robot_walk.mp4` (30fps, ~9초), `results/robot_walk.gif` (4초 루프)

## 환경

- Python 3.11
- numpy, alive-progress, noise, opencv-python-headless, mujoco, imageio
- JAX / PyTorch 불필요 (벤치마크 대상 코드는 순수 Python/numpy)
