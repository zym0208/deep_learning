# MotionRefiner 接口（Week 1 冻结）

这份文档是你和队友之间的约定。队友负责产出符合这个格式的 `.npz`，你负责 refiner 和 Unity。改格式要两个人一起改。

## 1. 动作文件格式（`.npz`，代码见 `refiner/motion_format.py`）

| key | shape | 单位 / 含义 |
|---|---|---|
| `right_pose` | (T, 48) | 弧度，MANO axis-angle。`[0:3]` 整手朝向，`[3:48]` 15 个手指关节 |
| `left_pose` | (T, 48) | 同上（左手按 S2C 惯例：用右手 MANO 解码后 x 取反） |
| `right_trans` | (T, 3) | 手腕平移，S2C 相机坐标（OpenCV：x 右、y 下、z 前），米 |
| `left_trans` | (T, 3) | 同上 |
| `fps` | 标量 | 30 |
| `valid_right` / `valid_left` | (T,) | 可选，标注有效 = 1。只在 GT 文件里有 |

约定：

- 时间：第 t 帧对应 `t / 30` 秒；8 秒窗口 = 240 帧。
- **存 S2C 平滑之前的值**（`infer.py` 里的 `right_pose_pred` 等，已经乘过 π 和 `[1.5, 1.5, 25]`）。
- 训练用一对文件：`<clip>_s2c.npz`（S2C 输出）和 `<clip>_gt.npz`（同一段音频的 PianoMotion10M 标注，用 `motion_format.from_annotation_json` 读）。
- 划分按视频（BV 号）分，不按帧随机分。初始目标 20 / 4 / 4 段。

## 2. Refiner 输入 / 输出

```python
from refiner import MotionRefiner
model = MotionRefiner()                     # 未训练时输出 == 输入（零初始化）
refined = model(batch)                      # batch: dict，每个 key (B, T, D)
refined, residual = model(batch, return_residual=True)
```

- **输入**：两只手的 `pose`(48) + `trans`(3)，共 102 通道。内部做归一化（pose / π，trans / [1.5, 1.5, 25]）。
- **网络**：Conv1d(k=5) → GELU → Dropout → Conv1d(k=5, dilation=2) → GELU → 1×1 输出头。感受野 13 帧（约 0.43 秒）。
- **输出头零初始化**：训练开始时 refiner 恒等于原始 S2C，任何改进都是学出来的。
- **输出**：
  - 15 个手指关节的旋转修正 `drot`，`tanh × max_rot`（默认 0.25 rad），用**旋转矩阵相乘** `R = R_raw · exp(drot)` 合成，不是 axis-angle 直接相加。
  - 手腕平移修正 `dtrans`，`tanh × max_trans`（默认 0.05，归一化单位，x/y 上约 7.5 cm）。
  - **不改**整手朝向 `pose[0:3]`（v1 的约定）。
- 输出与输入格式完全一样，可以直接交给 `tools/export_unity.py`。

## 3. 损失（`refiner/refiner_losses.py`）

| 项 | 作用 | 默认权重 |
|---|---|---|
| `rec_rot` | 手指旋转矩阵与 GT 的 MSE | 1.0 |
| `rec_trans` | 手腕位置 L1（归一化） | 1.0 |
| `vel_rot` / `vel_trans` | 相邻帧变化量与 GT 一致，保住快速击键 | 1.0 / 1.0 |
| `res_rot` / `res_trans` | 修正量的 L2，别离 S2C 太远 | 0.1 / 0.1 |

`valid == 0` 的帧不计入损失。权重是起点，Week 3 根据真实误差再调。

## 4. 三个对比条件（最终报告）

1. 原始 S2C（缓存的未平滑输出）
2. 原始 S2C + Savitzky-Golay(5, 2)，`refiner/baselines.py`
3. 原始 S2C + MotionRefiner

三者用同一份缓存、同一个导出脚本、同一个 Unity 场景。

## 5. Unity 文件（`.hmotion`）

由 `tools/export_unity.py` 生成，`unity/Assets/Scripts/HandMotionPlayer.cs` 读取。已经是 Unity 坐标（y 朝上、左手系、米），默认以平均手腕位置为原点。二进制布局写在 exporter 文件头注释里。
