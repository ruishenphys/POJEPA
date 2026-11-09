# PO-JEPA 可调物理基准（v7，v8 扩展中）

一个可生成、可调节的“画面 + 动作 + 真值物理状态”数据集。每条轨迹是 33 帧 256×256 RGB 画面（4 fps，0.25 s 一帧）、32 个 tick 的二维动作，以及生成画面用的隐藏物理状态。仿真数据的价值在于有真值方程和真值状态：可以检验模型学到的是不是正确的物理演化，这一点真实数据做不到。

数据集由三部分决定：**场景 × 调节项 × 条数**。写法是 `系统[~调节项]:条数`，例如：

| 写法 | 含义 |
|---|---|
| `double_well:2000` | v6 的双势阱（基础场景，与 v6 逐位一致） |
| `double_well~axes=x,gain=2:2000` | 控制增益 2，只能推 x 方向（欠驱动） |
| `landscape~shape=ring:2000` | 势能地形族里的环形槽 |
| `van_der_pol~background=checker,color=white,mu=1:2000` | 换物理参数和外观 |

调节项的完整定义和取值范围见 `src/po_jepa/variants.py`。仿真与渲染见 `src/po_jepa/systems.py`；v8 新加的 `gray_scott`、`push` 的物理与渲染在 `src/po_jepa/new_systems.py`（公式和参数范围的依据见 `docs/segments/S20a.md` §2）；五个二阶场景在 `src/po_jepa/second_order.py`（`docs/segments/S23a.md` §3，登记见 `S23b.md`）。

## 场景

前 7 个场景是一阶系统：画面里能看到决定未来的全部状态，因为 v7 的模型只看当前一帧（`occluders` 调节项有意破坏这一点，见下）。

后 5 个是二阶场景（S23a 物理和渲染，S23b 登记）：画面只给位置，速度要由相邻两帧给出。数据集照样是 33 帧（tick 0…32），编码、缓存、训练的入口不变；每条轨迹另外多仿真、多画 t = −1 一帧（初值在 t = −1 抽，零动作预跑一个 tick 到 t = 0），由 `systems.get().frames_with_prev()` 给出 34 帧，留给两帧（pair）编码。pair 编码怎么接要等 S22 两帧 tubelet 探针（`tools/tubelet_probe.py`，定义和判据见 `docs/segments/S22.md`）的结果；在那之前用单帧编码时，模型看不到速度。

| 场景 | 方程 | 画面 | 隐藏状态维数 |
|---|---|---|---|
| `double_well` | dx/dt = x − x³ − c·y + tilt + u_x，dy/dt = −y − c·x + u_y（c = 0.25） | 一个光点 | 2 |
| `van_der_pol` | dx/dt = v + u_x，dv/dt = μ(1 − x²)v − x + u_y（μ = 0.5） | 相平面上的一个光点 | 2 |
| `multi_particle` | n 个过阻尼粒子，两两软排斥加简谐外势，动作只推第 0 个（n = 3） | n 个彩色光点 | 2n |
| `advection_diffusion` | ∂c/∂t = D∇²c − w·∇c，w = flow·u（周期域，傅里叶精确解） | 浓度热图 | 256（16×16 池化） |
| `landscape` | dq/dt = −depth·∇V(q) + u，V 可选 4 种地形 | 光点和地形底图 | 2 |
| `gray_scott` | Gray–Scott 反应扩散 ∂U/∂t = D_u∇²U − UV² + F(1−U)，∂V/∂t = D_v∇²V + UV² − (F+k)V + a·G(x−p)(1−V)，注入点 dp/dt = 0.4·u（周期域） | U、V 两色叠加的浓度图和注入点圆环 | 514（U、V 各 16×16 池化，加注入点） |
| `push` | 准静态推方块：推杆 dc/dt = 2u − f，方块按椭圆极限面和库仑静摩擦阈值滑动、转动，可加坡面 | 方块（带朝向标记）和推杆光点 | 5 |
| `pendulum` | θ̈ = −(g/L) sin θ − γθ̇ + 4(u_x cos θ + u_y sin θ)（二维力取切向分量） | 支点、摆杆和摆锤光点 | 2（θ, θ̇） |
| `bouncing_balls` | 方盒里 n 个球，dv/dt = −βv + [i = 0]·1.5u，球与墙、球与球按恢复系数碰撞（事件驱动） | 方框和 n 个彩色圆盘 | 4n（位置、速度；n = 3） |
| `wave` | 周期域 u_tt = c²∇²u − γu_t + load·c²∇²B(x − p)，激振源 dp/dt = 0.4·u（傅里叶精确传播子） | 正负两色的位移场和激振源圆环 | 514（u、u_t 各 16×16 池化，加激振源） |
| `cart_pole` | 小车倒立摆的拉格朗日方程，只受水平推力 F = 1.5u_x（u_y 不起作用），轨道两端有缓冲 | 轨道、小车、摆杆和摆锤光点 | 4（x, ẋ, θ, θ̇） |
| `spring_chain` | 两端固定的线性弹簧链，d̈_i = −k(2d_i − d_{i−1} − d_{i+1}) − γḋ_i + [i = 0]·1.5u（矩阵指数精确推进） | 墙、弹簧线和 n 个彩色光点 | 4n（位置、速度；n = 4） |

`landscape` 的 4 种地形：

- `multi_well`：四个势阱。
- `ring`：环形槽。
- `periodic`：周期势阱阵列。
- `obstacles`：三个障碍物。

## 调节项

| 类别 | 名字 | 默认 | 范围 | 作用 |
|---|---|---|---|---|
| 物理 | `coupling`、`tilt` | 0.25、0 | [0, 0.6]、[−0.3, 0.3] | 双势阱耦合、倾斜 |
| 物理 | `mu` | 0.5 | [0.1, 1.5] | Van der Pol 非线性强度 |
| 物理 | `n`、`repulsion`、`trap` | 3、1.5、0.5 | 1–5、[0, 3]、[0.2, 1] | 粒子数、排斥强度、外势强度 |
| 物理 | `diffusion`、`flow`、`blobs` | 2e-4、0.4、3 | [0, 2e-3]、[0.1, 1]、1–5 | 扩散系数、流速比例、初始染料团数 |
| 物理 | `shape`、`depth` | —、1 | 4 种、[0.25, 3] | 地形形状、地形深度 |
| 物理 | `feed`、`kill`、`inject` | 0.04、0.06、0.05 | [0.035, 0.05]、[0.0595, 0.063]、[0, 0.2] | Gray–Scott 补给率、消耗率、注入强度 |
| 物理 | `mu_s`、`slope`、`slope_dir`、`size` | 0.3、0、0、0.8 | [0, 1]、[0, 0.6]、[−180, 180] 度、[0.4, 1.2] | 推方块的静摩擦系数、坡度、坡向、方块边长 |
| 物理 | `g_over_l`、`damping` | 9.81、0.2 | [4, 16] s⁻²、[0, 1] s⁻¹ | 单摆 g/L、阻尼 |
| 物理 | `n`、`restitution`、`drag` | 3、0.9、0.2 | 1–5、[0.5, 1]、[0, 1] s⁻¹ | 弹球个数、恢复系数、线性阻力 |
| 物理 | `speed`、`damping`、`load` | 0.15、0.1、0.25 | [0.08, 0.3] 域长/秒、[0, 0.5] s⁻¹、[0, 0.4] | 波速、阻尼、激振源压下的深度 |
| 物理 | `length`、`mass_ratio`、`friction` | 1、0.3、0.5 | [0.6, 1.4]、[0.1, 1]、[0, 1] | 倒立摆杆长、摆锤与小车质量比、小车摩擦 |
| 物理 | `n`、`stiffness`、`damping` | 4、16、0.1 | 2–6、[9, 25] s⁻²、[0, 0.5] s⁻¹ | 弹簧链质点数、劲度、阻尼 |
| 控制 | `gain` | 1 | [0.25, 4] | 动作强度（作用在物理上的倍数） |
| 控制 | `axes` | xy | xy / x / y | 只能推一个方向（欠驱动） |
| 控制 | `segments` | 148 | 148 / 1 / 2 / 4 / 8 | 动作切换长度：v6 的 1/4/8 随机，或固定周期 |
| 控制 | `zero` | 0.2 | [0, 0.8] | 零动作段比例 |
| 控制 | `noise` | 0 | [0, 0.5] | 执行噪声（模型看不到） |
| 控制 | `delay` | 0 | 0–4 tick | 执行延迟（模型看不到） |
| 控制 | `saturation` | 无 | [0.05, 2] | 执行动作按分量截断到 [−s, s]，在增益和执行噪声之后（模型看不到） |
| 数据采集 | `policy` | random | random / goal / mixed | 采集策略：random 是 v6 的随机动作；goal 朝目标推（粒子类场景的目标在 [−1.5, 1.5]² 随机，`push` 是穿过方块中心往前 0.6）；mixed 每个非零段一半随机一半朝目标。朝目标是开环推算，不跑仿真 |
| 物理 | `process_noise` | 0 | [0, 0.3] | 过程噪声 σ（状态单位/√秒）：dq = f dt + σ dW，只用于粒子类 ODE 场景 |
| 观察 | `sigma` | 场景默认 | [3, 16] 像素 | 光点大小（`advection_diffusion`、`gray_scott`、`wave` 没有光点，`bouncing_balls` 的球是有物理半径的圆盘，都不适用） |
| 观察 | `color` | orange | orange / white / green / blue | 受控光点颜色 |
| 观察 | `background` | grid | grid / plain / checker / noise | 背景 |
| 观察 | `distractors` | 0 | 0–6 | 静止干扰光点（`advection_diffusion`、`gray_scott`、`wave` 不适用） |
| 观察 | `movers` | 0 | 0–4 | 会动的灰色干扰光点，绕画面中心以 0.6 rad/s 逆时针转，半径 40–110 像素；只看一帧就能推出下一帧（`advection_diffusion`、`gray_scott`、`wave` 不适用） |
| 观察 | `occluders` | 0 | 0–3 | 静止不透明遮挡块（48×48 像素，每条轨迹位置不同），画在场景之上：被遮住时状态看不到（部分可观测） |
| 观察 | `pixel_noise` | 0 | [0, 40] | 每帧像素噪声 |

规则：

- **配对**：调节项不改变初始状态和动作的随机种子；只有改动作抽样的 `axes`、`segments`、`zero`、`policy` 例外。所以同一编号的轨迹在不同调节项下可以逐条配对比较。`policy` 只换非零段的取值，段的边界和零动作段与 random 相同。
- **随机流**：每条轨迹按用途分流，0 初值、1 动作、2 预处理、3 静止干扰光点、4 像素噪声、5 执行噪声、6 过程噪声、7 遮挡块、8 会动的干扰光点、9 采集策略。加调节项只会用到新的流，不改动已有流。
- **反事实**：过程噪声按“轨迹 + 绝对 tick”预先抽好，从任意 tick 出发的反事实仿真用同一份噪声，只换动作。
- **基础场景不变**：调节项为空时走原实现，与 v6 和 v7 试跑逐位一致，有单元测试保证。只改观察的调节项，隐藏状态也和基础场景逐位相同。
- **软墙**：可调版本在画面边缘有一圈软墙，增益或参数调大时小球也不会跑出画面。基础场景不加软墙。
- **新场景**：`gray_scott`、`push` 没有 v6 原实现要保持，基础场景和有调节项时走同一套代码。它们的随机流序号是 5、6（追加在 `datasets.SYSTEMS` 末尾），v7 的 5 个场景的随机流不变。二阶场景同样，序号 7–11（`pendulum`、`bouncing_balls`、`wave`、`cart_pole`、`spring_chain`）。

### 覆盖表

每个通用调节项在哪些场景可用，由 `tools/knob_coverage.py` 从 `variants.validate` 的判定生成（测试核对本表与代码一致）。

<!-- coverage:begin（由 tools/knob_coverage.py 生成，不要手改）-->
| 类别 | 调节项 | `double_well` | `van_der_pol` | `multi_particle` | `advection_diffusion` | `landscape` | `gray_scott` | `push` | `pendulum` | `bouncing_balls` | `wave` | `cart_pole` | `spring_chain` | 可用场景数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 控制 | `gain` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `axes` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `segments` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `zero` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `noise` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `delay` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 控制 | `saturation` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 数据采集 | `policy` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | — | — | — | — | — | 5 |
| 物理 | `process_noise` | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | — | — | — | 4 |
| 观察 | `sigma` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | — | — | ✓ | ✓ | 8 |
| 观察 | `color` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 观察 | `background` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 观察 | `distractors` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | ✓ | — | ✓ | ✓ | 9 |
| 观察 | `movers` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | ✓ | — | ✓ | ✓ | 9 |
| 观察 | `occluders` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |
| 观察 | `pixel_noise` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 12 |

通用调节项 16 个，“调节项 × 场景”可用格 167 / 192。物理参数（各场景自己的）见上表。
<!-- coverage:end -->

## 任务

| 任务 | 做法 | 命令 |
|---|---|---|
| 前向预测 | 给当前帧、动作和时长，预测未来帧的特征。在时长留出（3、12 周期）、网格留出（8×8、12×12）和二者同时留出上评测 | `run.py run` |
| 自洽性 | 时间组合缺陷 TCD、跨网格缺陷 SCD | `run.py run` |
| 动作敏感性 | 换成另一条轨迹的动作后，误差应该变大多少（delta_act），并与完美模型的上界比较 | `run.py run`，`tools/probe_actions.py` |
| 分布外泛化 | 在基础场景上训练，在换了物理参数、控制方式或外观的测试集上评测（按下面的评测协议） | `run.py transfer` |
| 规划 | 给目标画面，用模型和 CEM 搜动作，把受控物体推到目标；报成功率，并给不动、随机、真值规划三个参照 | `tools/plan_v7.py` |

前 7 个场景都能跑动作探针和规划（S21b 接上了 `gray_scott`、`push`）；二阶场景还没接（两个工具对它们直接报错），要等两帧编码定下来。规划的“状态误差”按场景取：小球场景是受控小球位置，
`advection_diffusion` 是 16×16 平均浓度场，`gray_scott` 是 U、V 的 16×16 池化场，`push` 是方块位姿（推杆在哪不算）。
`push` 的目标由“朝方块推”的动作生成（随机动作很少碰到方块），真值规划的代价用推杆加方块的完整状态。细节见 `tools/plan_v7.py` 文件头。

## 评测协议

训练只在基础场景上（`landscape` 取多势阱地形）。测试按协议换调节项，每个协议在基础场景测试段同一批 200 条轨迹上评
（变体测试集的编号与基础场景测试段一一对应，初始随机抽样相同；`gray_scott` 的第 0 帧是按当前物理参数烧入后的状态，换 feed、kill 时第 0 帧的 U、V 也会不同，只有注入点相同），写法参照 CausalWorld：每个协议只动一类变量，P13 把物理、执行、
观察三类叠在一起。P0 是分布内（基础场景测试段本身）。强度分轻、标准、重三档：分布外评测
（`tools/make_v8_jobs.py --stage variants`）只评标准档，强度曲线（`--stage strength`）评轻档和重档。规划只在 P0 上做。

v7 的分布外测试表（`tools/make_v7_jobs.py` 的 `TEST_VARIANTS`，30 个）正好是这里对应协议在 v7 五个场景上的标准档，
v7 的表本身不改（v7 正式实验和仿真补跑都在用）。协议的唯一来源是 `src/po_jepa/protocols.py`，下表由
`tools/protocol_table.py --write` 生成。

<!-- protocols:begin（由 tools/protocol_table.py 生成，不要手改）-->
| 协议 | 类别 | 变什么 | 轻 | 标准 | 重 | 不适用的场景 |
|---|---|---|---|---|---|---|
| P1 物理参数 A | 物理 | 每个场景自己的第一个物理参数调大 | 见下表 | 见下表 | 见下表 | — |
| P2 物理参数 B | 物理 | 第二个物理参数（只有一个参数的场景把它往另一边调） | 见下表 | 见下表 | 见下表 | — |
| P3 过程噪声 | 物理 | dq = f dt + σ dW（只用于粒子类 ODE 场景） | `process_noise=0.05` | `process_noise=0.1` | `process_noise=0.2` | `advection_diffusion`、`gray_scott`、`push` |
| P4 执行增益 | 执行 | 执行动作 = 增益 × 下达动作（模型看不到） | `gain=1.5` | `gain=2` | `gain=3` | — |
| P5 执行延迟 | 执行 | 执行动作晚若干 tick（模型看不到） | — | `delay=1` | `delay=2` | — |
| P6 执行饱和 | 执行 | 执行动作按分量截断到 [−s, s]（模型看不到） | `saturation=0.4` | `saturation=0.3` | `saturation=0.15` | — |
| P7 采集策略 | 动作分布 | 测试轨迹的动作换成朝目标（goal）或一半朝目标（mixed） | `policy=mixed` | `policy=goal` | — | `advection_diffusion`、`gray_scott` |
| P8 外观 | 观察 | 背景和受控物体颜色 | `color=white` | `background=checker,color=white` | `background=noise,color=blue` | — |
| P9 静止干扰光点 | 观察 | 灰色干扰光点，不受控制 | `distractors=1` | `distractors=2` | `distractors=4` | `advection_diffusion`、`gray_scott` |
| P10 会动的干扰光点 | 观察 | 绕画面中心匀速转的灰色光点 | `movers=1` | `movers=2` | `movers=4` | `advection_diffusion`、`gray_scott` |
| P11 像素噪声 | 观察 | 每帧独立的高斯像素噪声（0–255 刻度） | `pixel_noise=5` | `pixel_noise=10` | `pixel_noise=20` | — |
| P12 遮挡 | 观察 | 48×48 像素的静止遮挡块（部分可观测） | `occluders=1` | `occluders=2` | `occluders=3` | — |
| P13 组合 | 组合 | P1、P4、P8 的同一档叠加 | 见下表 | 见下表 | 见下表 | — |

P1、P2 各场景的档位（P13 用 P1 的同一档）：

| 场景 | P1 轻 | P1 标准 | P1 重 | P2 轻 | P2 标准 | P2 重 |
|---|---|---|---|---|---|---|
| `double_well` | `coupling=0.4` | `coupling=0.5` | `coupling=0.6` | `tilt=0.1` | `tilt=0.2` | `tilt=0.3` |
| `van_der_pol` | `mu=0.8` | `mu=1` | `mu=1.5` | `mu=0.35` | `mu=0.2` | `mu=0.1` |
| `multi_particle` | `repulsion=2` | `repulsion=2.5` | `repulsion=3` | `trap=0.65` | `trap=0.8` | `trap=1` |
| `advection_diffusion` | `diffusion=0.0005` | `diffusion=0.001` | `diffusion=0.002` | `flow=0.6` | `flow=0.8` | `flow=1` |
| `landscape` | `depth=1.5` | `depth=2` | `depth=3` | `depth=0.75` | `depth=0.5` | `depth=0.25` |
| `gray_scott` | `feed=0.043` | `feed=0.046` | `feed=0.05` | `kill=0.061` | `kill=0.062` | `kill=0.063` |
| `push` | `mu_s=0.45` | `mu_s=0.6` | `mu_s=0.9` | `slope=0.1` | `slope=0.2` | `slope=0.25` |

适用的测试格（7 个场景合计）：标准档 82 个，轻档 75 个，重档 77 个。P0 分布内就是基础场景的测试段，不另编码。
<!-- protocols:end -->

## 生成与划分

- **划分**：按编号连续切，80% 训练、10% 验证、10% 测试，单位是完整轨迹。
- **条数**：可选 40、200、500、2000。
- **变体测试集**：只编码测试段（`encode_v7.py --splits test`），编号和基础场景的测试段一一对应。
- **编码**：冻结的 V-JEPA 2 ViT-g/16，权重 SHA 67129f…，上游提交 204698b，和 v3 起一直用的是同一套预处理。特征存 float16，每条约 24 MB。
- **可追溯**：每条轨迹的清单里记录场景、调节项、动作、每帧画面的 SHA、随机种子和生成代码的 SHA。

## 已知局限

- 模型只看单帧；二阶场景已登记，但两帧（pair）编码要等 S22 探针结果，单帧编码时看不到速度。
- `process_noise` 只用于粒子类 ODE 场景；`policy` 不用于场方程（`advection_diffusion` 的动作是整场流速，`gray_scott` 是注入点速度，没有要推到目标的受控点），也不用于二阶场景（动作是力，`policy=goal` 的开环推算按“动作 ∝ 受控点速度”做，对它们不成立）。
- `cart_pole` 只受水平推力，`axes=y` 等于没有控制，不接受。
- `policy=goal` 是开环推算，不知道执行时的增益、延迟、噪声、饱和和场景的动力学，所以并不总能到目标；它的作用是让动作更多地作用在受控物体上（`push` 上推动方块的轨迹比例明显高于 random，实测见 `docs/segments/S21a.md` §4）。
- `gray_scott`、`push` 在 v8 的任务表（`tools/make_v8_jobs.py`）里，不在 v7 的任务表里。它们的真值规划参照每个 tick 的仿真比粒子类场景贵 10–20 倍，v8 任务表把它们的 `--oracle-samples` 降到 16（`gray_scott`）和 32（`push`）。
- 只画部分帧时（动作探针、规划），像素噪声按 tick 取整条轨迹同一份噪声（S21b 起）；以前单帧重画用的是第 0 帧的噪声。
- `push` 在 v6 的随机动作下，大约三分之一的轨迹真正推动了方块，动作的作用偏弱。
- 动作统一是二维，范围 [−0.5, 0.5]²。
- 画面是合成的简单图形，和真实视频有差距。真实数据（DROID）放在下一阶段。
