# Week 1：S2C `infer.py` 是怎么把音频变成双手动作的

对应代码：<https://github.com/monkek123King/S2C>（`infer.py`、`models/piano2posi.py`、`models/denoise_diffusion.py`、`datasets/PianoPose.py`、`datasets/show.py`）。

## 一句话

**音频 → HuBERT 特征 → 位置预测器（Piano2Posi）给出两只手腕的 3D 轨迹 → 双流扩散模型（左右手各一个 UNet，互相做 cross-attention）以音频特征和手腕轨迹为条件，去噪出每帧 48 维 MANO 姿态 → MANO 解码成网格。**

## 数据长什么样（PianoMotion10M）

一个 annotation JSON 里 `right` / `left` 每一帧是一行 62 个数：

| 列 | 含义 |
|---|---|
| 0 | 有效标记（1 = 这一帧有标注；训练时用作 mask） |
| 1:4 | 手腕平移 (tx, ty, tz)，相机坐标系（OpenCV：x 右、y 下、z 朝前），单位米 |
| 4:7 | global orient（整只手的朝向），axis-angle |
| 7:52 | 15 个手指关节，每个 3 维 axis-angle，共 45 维 |
| 52:62 | MANO betas（手型），`return_beta=False` 时被丢掉 |

`--adjust` 打开时，`PianoPose.py` 会把 `mano['adjust'][1:3]` 加到 (ty, tz) 上。音频是 16 kHz，动作重采样到 30 fps，每段训练窗口 `train_sec=8` 秒 = 240 帧。

## infer.py 一步步

1. **读配置**：从 `exp_path/args.txt` 和 `args_posi.txt` 读超参，`up_list` 被写死成两个 UP 主（`1467634`, `66685747`）。
2. **建模型**：
   - `Piano2Posi`：HuBERT-large 提 1024 维音频特征 → Transformer 编码/解码 → `tanh` 输出每帧 3 维手腕位置（左右各一个模型）。
   - `Unet1D` + `GaussianDiffusion1D_piano2pose`：1D UNet，输入 48 通道（一只手的姿态），条件是音频特征（`cond`）和手腕位置（`guide`，6 维 = 本手 3 + 另一只手 3）。左右手两个 UNet 在去噪时通过 `CrossAttentionFusion` 交换信息，这就是论文里 "Separate to Collaborate" 的双流。
3. **加载权重**：`ckpt` 文件名里必须同时包含 `left`/`right` 和 **`609000`**（写死的 step），否则找不到。
4. **取一段数据**：`test_ids = np.arange(0, 1)` 写死只跑第 0 段。注意 `getGuideAndCond(gt_left, audio)` 里 GT 只用来拿帧数 `n`，**没有泄漏 GT 动作**。
5. **采样**：`model_left.sample(...)` 内部是 `p_sample_loop`，两只手从同一份高斯噪声出发，逐步去噪 `timesteps` 步（`pred_v` 目标）。每一步左手 UNet 和右手 UNet 互相看对方的中间特征。
6. **还原尺度**：
   - 姿态：网络输出在 [-1, 1] 左右，`× π` 得到弧度。
   - 平移：`guide × [1.5, 1.5, 25]` 得到相机坐标系的米。
7. **平滑**：对姿态和平移各做一次 Savitzky-Golay（窗口 5、阶数 2）。
8. **指标与渲染**：算 FID 和手腕位置误差，然后用 `nvdiffrast` 在 GPU 上逐帧渲染 PNG。左手是用**右手 MANO 模型**算网格再把 x 取反（镜像）得到的。

## 对我们有影响的坑

| 问题 | 位置 | 怎么处理 |
|---|---|---|
| `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` 写死 | infer.py 第 5 行 | 在超算上会覆盖 Slurm 分配的 GPU，**删掉** |
| ckpt 过滤写死 `'609000'` | 加载权重处 | 改成命令行参数 `--step` |
| `test_ids` 写死 | 采样循环 | 改成参数，或循环整个 split |
| 渲染需要 nvdiffrast + OpenGL | `datasets/show.py` 顶部 `RasterizeGLContext()` | headless 节点上会崩；**只存数组，不 import show.py** |
| "原始 S2C" 已经被平滑过 | 第 7 步 | 缓存时存**平滑前**的 `right_pose_pred / left_pose_pred / right_trans_pred / left_trans_pred`，平滑作为单独的 baseline |
| 自定义音频没有入口 | 只能从 dataset 取 | 队友 Week 2 写 `infer_audio.py`：librosa 16 kHz 读入、切 8 秒窗口 |
| README 写 `requirement.txt` | 实际文件是 `requirements.txt` | — |

## refiner 接在哪

```
infer.py 第 6 步之后、第 7 步之前
   right_pose_pred, left_pose_pred   (T, 48) 弧度
   right_trans_pred, left_trans_pred (T, 3)  米
        │  存成 .npz（格式见 refiner_interface.md）
        ▼
   MotionRefiner（我们训练）  ──► 导出 .hmotion ──► Unity
```

S2C 全程冻结；refiner 只读缓存好的预测，不需要每个 batch 重跑扩散采样，也不对 S2C 反向传播。
