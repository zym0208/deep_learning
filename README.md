# 钢琴双手动作生成（课程作业）

MIDI / 乐谱 → 合成音频 → **冻结的预训练 S2C 扩散模型** → **我们训练的 MotionRefiner** → Unity 实时播放。

## 目录

| 路径 | 内容 | 负责人 |
|---|---|---|
| `docs/week1_s2c_infer_walkthrough.md` | S2C `infer.py` 从音频到双手动作的流程，以及要改的坑 | yumin |
| `docs/refiner_interface.md` | 动作 `.npz` 格式、refiner 输入输出、损失、对比条件（**双方约定**） | yumin + 队友 |
| `refiner/motion_refiner.py` | 时序残差网络（零初始化、有界修正、旋转相乘） | yumin |
| `refiner/refiner_losses.py` | 重建 + 速度 + 修正量损失，支持有效帧 mask | yumin |
| `refiner/rotation.py` | axis-angle ↔ 旋转矩阵 | yumin |
| `refiner/motion_format.py` | 读写 `.npz`，读 PianoMotion10M 标注 | 共用 |
| `refiner/baselines.py` | Savitzky-Golay 平滑 baseline | 共用 |
| `tools/export_unity.py` | 动作 → `.hmotion`（MANO 网格，Unity 坐标） | yumin |
| `unity/Assets/Scripts/HandMotionPlayer.cs` | Unity 播放器 | yumin |
| `tests/` | Week 2 验收测试 | yumin |
| `infer_audio.py`、`cache_predictions.py`、`train_refiner.py` | 尚未开始 | 队友 |

## 快速开始

```bash
pip install -r requirements.txt
pytest -q                                          # 17 个测试，CPU 即可
python tools/export_unity.py --dummy -o unity/Assets/StreamingAssets/dummy.hmotion
```

### Unity（Week 1 验收）

1. 用 Unity Hub 打开（或新建）一个 3D 项目，把本仓库 `unity/Assets/` 下的 `Scripts/` 和 `StreamingAssets/` 拷进项目的 `Assets/`。
2. 新建空物体，挂 `HandMotionPlayer`，`File Name` 填 `dummy.hmotion`，点 Play。应看到两只方块手轮流按下手指。
3. 拿到 MANO 和真实数据后，换成真实网格：

```bash
# PianoMotion10M 的标注（不用跑 S2C，可以先验证坐标）
python tools/export_unity.py --annotation <data>/annotation/<up>/<bv>/<bv>_seq_0000.json \
       --mano-root <放 mano/MANO_RIGHT.pkl 的目录> -o unity/Assets/StreamingAssets/gt.hmotion
# 队友缓存好的 S2C 输出
python tools/export_unity.py --clip cache/<clip>_s2c.npz --mano-root <...> -o unity/Assets/StreamingAssets/s2c.hmotion
```

空格 播放/暂停，R 重播，←/→ 暂停时逐帧。手看起来"里外翻"时勾选 `Flip Winding`。挂一个 AudioSource 并拖到 `Audio` 字段，播放时钟就以音频为准。

### Week 3：原始 / 修正版本切换和误差分析

- Unity：在 `HandMotionPlayer` 的 `Variant Files` 里依次填 `raw.hmotion`、`smooth.hmotion`、`refined.hmotion`（同一段、同样帧数）。播放中按 1/2/3（或 Tab）切换版本，不打断音乐，停在同一帧。音频按采样计时（`timeSamples`），`Audio Offset` 可微调手和声音的先后。
- 分析：

```bash
python tools/analyze_motion.py cache/<clip>_s2c.npz cache/<clip>_refined.npz --smooth \
       --gt cache/<clip>_gt.npz --plot week3_<clip>.png
```

打印每只手的手腕抖动（jerk）、手指角加速度、8 Hz 以上能量占比、95% 手指速度（快速触键是否被抹平），有 GT 时还有手腕误差、漂移斜率（mm/s）和手指角度误差，并列出抖动最严重的帧，方便在 Unity 里逐帧看。

## 现状（2026-11-12）

- Week 1（yumin）：infer 流程梳理 ✓、接口约定 ✓、Unity 播放器 ✓（dummy 文件；真实 MANO 网格需要 MANO_RIGHT.pkl，尚未在 Unity 里实测）。
- Week 2（yumin）：refiner、损失、梯度 / 零初始化 / 存取 / 单 batch 过拟合测试 ✓（合成数据，CPU）。
- Week 3（yumin，不需要数据的部分）：Unity 版本切换 + 音画同步 ✓（未在 Unity 里编译实测）、抖动 / 漂移分析脚本 ✓。调损失和残差上限要等真实 S2C 预测。
- Week 3 分析和第一次小训练（Parnia 的 28 段缓存）：见 `docs/week3_analysis.md`。结论是过拟合，需要更多数据。
- 依赖队友：跑通 S2C 官方示例、缓存预测与标注、`train_refiner.py`。
