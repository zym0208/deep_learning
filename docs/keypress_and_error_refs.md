# 按键检查（MIDI）和误差参考值（2026-11-27）

对应教练 Juha 18/11 的第 2、3、4 条：用论文里的误差做参照、在 Unity 里显示按键、检查按键位置对不对。

## 1. 按键：从 MIDI 拟合键盘，再看指尖在不在键上

PianoMotion10M 每段都有 MIDI（`<data_root>/midi/<up>/<video>/<seq>.mid`），但没有给键盘在相机里的位置。
每个音符的起音时刻都等于说"此刻有一根手指在第 k 个键上"，所以键盘位置可以从数据里拟合出来：

- 键 k 的位置 = o + u(k)·v，u(k) 是以白键为单位的坐标（黑键在两个白键中间），v 是一个白键宽度的向量。
- 不知道是哪根手指弹的，所以交替做：每个起音取离键最近的指尖 → 最小二乘重拟 o、v → 去掉偏差超过 2 个键的点，重复（一维 ICP）。手指之间正好约一个键宽，容易整体错一根手指，所以最后在 ±1 键平移、±10% 缩放上再网格搜一次。
- 用图像平面坐标（x/z, y/z），因为 S2C 的深度 z 不可信（见 `week3_analysis.md`）。距离以键宽为单位，再乘真实白键宽 23.5 mm 换成 mm，这样不受相机尺度影响。

指标（`refiner/keypress.py`，`tools/keypress_eval.py eval`）：

| 名称 | 含义 |
|---|---|
| key_hit_rate | 起音时刻，最近的指尖离正确键中心不超过半个键（沿键盘方向）的比例 |
| key_dist_mm | 最近指尖到键中心的平均距离，mm |

做法：每段在 **GT** 上拟合键盘，再用同一个键盘给 GT、原始 S2C、微调后的 S2C、第二个模型打分。GT 自己的命中率说明标注本身有多准，是任何模型的上限。
这个距离也可以直接当损失用（教练说的 "possible loss"）：起音帧上，最近指尖到键中心距离的平方。

Unity：`MidiKeyboard.cs` 用同样的方法在手的网格上拟合键盘。音符响时琴键按下并着色：橙 = 右手在键上，蓝 = 左手，红 = 没有指尖在这个键上（模型弹错了）。左上角显示每个版本（gt / s2c / refined）的命中率，按 1/2/3 切换版本时对比。

```bash
# HPC 上（需要 smplx + MANO）
python tools/keypress_eval.py eval cache/ --data-root /scratch/leuven/395/vsc39592/pianomotion \
       --mano-root <放 mano/MANO_RIGHT.pkl 的目录> --variants s2c --json keypress.json
# 给 Unity 的音符文件（不需要 MANO），放到 StreamingAssets/<clip>_notes.json 就会自动出现键盘
python tools/keypress_eval.py notes <data_root>/midi/<up>/<video>/<clip>.mid --duration 8 \
       -o <clip>_notes.json
```

还没在真实数据上跑过，只用合成数据测过（`tests/test_keypress.py`）。

## 2. 误差参考值：哪些能比，哪些不能

Parnia 说得对：S2C 和 PianoMotion10M 论文里的指标（FID、FGD、WGD、PD、Smooth 等）大多没有单位，是分布距离或者归一化后的数，我们的 8.62° 不能直接和它们比。
能比的是"测量本身有多准"的文献：如果我们的误差和动捕设备/姿态估计的误差同一个量级，就说明在标注噪声范围内。

| 来源（Parnia 找的，链接见下） | 数值 | 和我们哪个数比 |
|---|---|---|
| Leap Motion vs 动捕 | 关节角约 10° | 手指角度误差 8.62°（验证集） |
| Sensors 2021，表 5 | 每个关节角 3–12° | 同上 |
| SKY-Piano 论文里引用的 Tipiano | MPJPE 32.8 mm，按键接触 F1 0.910；视频数据有厘米级噪声 | MPJPE（要补算）；key_hit_rate |
| 3D 手部姿态估计挑战（arXiv 2003.13764） | 27 → 13 mm | MPJPE（要补算） |

要注意的三点：

1. **角度定义不同。** 我们的 8.62° 是 MANO 每个关节局部旋转的测地角误差（三个自由度合在一起）；那些文献多是单个弯曲角（测角仪那种）。量级可以比，数值不能严格等同。报告里写"same order as"，不要写"better than"。
2. **我们的位置误差现在不能和 MPJPE 比。** 现有的 27 mm 只是手腕、只有 x/y。要补一个 21 个关节、以手腕对齐（root-relative）的 MPJPE（mm），才能和 13–33 mm 放在一起。这个用 MANO 关节就能算，和按键指标一起在 HPC 上跑。
3. **我们的"真值"本身就是视频估计的。** PianoMotion10M 的标注来自视频，Tipiano/SKY-Piano 也提到厘米级噪声。所以低于约 1 cm / 几度的差别没有意义。按键命中率正好补上这一点：它不看 GT 的姿态，只看 MIDI，MIDI 是准的。

Tipiano 的数字是从 SKY-Piano 论文转引的，Parnia 还没打开 Tipiano 原文；直接引用前要查原文。我也没有逐个核对这些链接。

- https://www.iris.unicampus.it/retrieve/12d438c4-fbec-451c-ac03-a6649e78eb28/Performance_Evaluation_of_Leap_Motion_Controller_for_Hand_Motion_Assessment.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8659816/
- https://arxiv.org/pdf/2607.27296
- https://arxiv.org/abs/2003.13764
