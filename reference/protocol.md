# 第一轮受控视觉仿真协议：overdamped_double_well_v1

这是**结果产生前的协议**，不是实验结果。唯一可读配置是 [first_simulation.yaml](../first_round/configs/first_simulation.yaml)；本文件解释每个板块的目的和验收边界。配置不能直接当作 E06 的真实训练批次或启动命令。F01 只做静态核验，不生成轨迹、不编码、不训练。

## 1. 依据、选择与输入边界

原稿基准为 `docs/source/main.tex`（SHA-256 `d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1`），上游固定 `204698b45b3712590f06245fbfba32d3be539812`，主干基线 `01ffa7d458b3b07c67d552f766692ef265551209`。逐条 prompt 来自 Physics Operator Prompt，turn `01a14f8f-7461-72d3-84ca-1f5ff26e4d53`，完整原文和来源元数据随验收归档。

原稿规定预测模型、动作条件动力学和三项损失；**下面的具体仿真、渲染、数据数量和运行预算是实现选择**，不是导师原稿唯一指定的实验。沿单卡修订保留完整预测器与双一致性；完整 E08、DROID/AC/规划和额外消融仍后移。训练对照不能据此替代论文全部实验。

A05 锁定：预测起点只见一张当前 RGB 图，其两个相同副本用于 tubelet=2，不能算两帧运动历史。预测器输入为 raw Z、给定控制区间、查询时长和已知网格；Transformer 也得到同样信息。隐藏 q、速度、真实未来状态/图像/latent 不进入预测分支。未来 raw Z 只作监督目标。缓存只保存冻结 encoder 的 raw Z，不固定 X/W/L/谱/C/g；原稿式15的第二段从预测 Z 重新做完整 Phi，不继承隐藏状态。

## 2. 唯一任务与状态可观测性

任务 `overdamped_double_well_v1` 是有控制的二维一阶系统，q=(x,y) 为无量纲“模拟位置”，t 为物理秒：

```text
dx/dt = x - x^3 - 0.25*y + u_x
dy/dt = -y - 0.25*x + u_y
```

系数固定：x 的线性项 1、三次项 -1、交叉项 -0.25；y 的线性项 -1、交叉项 -0.25。控制两通道均在 [-0.5,0.5]，单位为模拟位置/秒。模型不会获得上述方程或系数、q 数组。该系统有非线性、耦合和动作切换；不把真值设成模型的线性阻尼旋转块，也不要求真流处处收缩。

状态域为 [-2,2]²。最坏合法控制下：
- x=2：6 的向内漂移被 y 与控制最多抵消 1，因此 dx/dt≤-5；
- x=-2：dx/dt≥5；
- y=2：dy/dt≤-1；
- y=-2：dy/dt≥1。

向量场光滑且边界向内，域内连续轨迹不需要反射、周期包裹或裁剪。数值轨迹越界要报错并追溯积分/实现，禁止 clamp 修改标签。当前位置 q 已是完整状态，没有隐藏速度或碰撞记忆；这仅解释**状态层面的充分性**。有限像素、裁剪和冻结 encoder 可能混淆状态；F02 必须实际诊断，不能宣称视觉 latent 已全局 Markov 充分。

## 3. 图像怎样只表达位置

每帧是 CPU uint8 的 [256,256,3] RGB，范围 0—255，固定相机、纹理、颜色，无噪声、遮挡、轨迹编号、时间、速度箭头、目标或状态文字。亮斑中心列为 128+48*x、行为 128-48*y；像素中心取 (row+0.5,col+0.5)。

版本 `soft_spot_rgb_v1`：普通背景 RGB=(24,28,32)，整数 row 或 col 能被 32 整除时为宽 1 像素的固定网格线，RGB=(36,40,44)。亮斑 RGB=(230,150,50)、sigma=8 像素。对每个像素先设背景，再计算：

```text
w = exp(-((row+0.5-center_row)^2+(col+0.5-center_col)^2)/(2*8^2))
RGB = numpy.rint(background + (color-background)*w).astype(uint8)
```

舍入为最近偶数（ties to even），固定实现版本与 NumPy 版本；不按样本改变。参考网格只提供静态纹理。q 和积分误差只用于生成/诊断，不是模型输入或辅助 loss。本步不生成图像。

## 4. 冻结编码及同轨迹裁剪

沿 B03 的 `FrozenFrameEncoder`：`vit_giant_xformers`，输入 256、patch=16、tubelet=2、E=1408、一个实际时间块、16×16 空间 token。encoder_key=`target_encoder`，既有来源元数据 checkpoint SHA 为 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`。这里只引用已验收来源，**没有重新读取权重，也不表示实际路径当前可用**；运行者显式提供路径并核验。保持冻结、eval、autocast=None，无 LoRA、EMA 或新归一化。

B03 输入范围 [0,255]，crop_size=256、scale=(1.777,1.777)、ratio=(0.75,1.35)、mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，不改已验收预处理。每条轨迹的每一帧都先把独立 CPU generator 重置到该轨迹的同一个初始状态，**逐帧**预处理，然后才允许小 batch 编码。同种子不能代替同映射的证明：F02 要记录并核对实际裁剪映射、亮斑可见性；256 方图可能触发中心回退，但不能预先判定为恒等。

每条轨迹 raw Z 形状为 [33,256,1408]、float32、无梯度。未来端点独立渲染并用冻结 encoder 编码；不能用模型预测、latent 插值或隐藏状态标签制造未来目标。

## 5. 秒、控制周期与参考积分

`ControlSpec` 的 action_dim=2，channel_names=(u_x,u_y)，channel_units 均为 simulation_position_per_second；lower_bounds=(-0.5,-0.5)、upper_bounds=(0.5,0.5)、scale=(1,1)、offset=(0,0)，definition_source 指本文件。动作是分段常值控制命令，不是整段位姿增量；切分区间不缩放或复制整段增量。

p=0.25 秒/控制周期；内部时间 s=(t_seconds-origin_seconds)/p，origin=0 秒。一周期等于内部时间 1。真值积分用秒，预测器用控制周期，幅值不再乘除 p；不能把时间转换重复应用。每条轨迹 32 周期=8 秒，观测 tick 为整数 0…32，共33帧。时间按整数 tick×0.25 构造，禁止累计浮点加时钟。

参考生成器是独立 NumPy float64 固定步 RK4，不调用模型 exponential-Euler 产生真值。每周期128步，h=0.001953125秒；半步0.0009765625秒。在一个动作段的最后一步，四个 RK4 stage 都用该段固定命令，下一步才用右侧新命令，避免末 stage 提前切动作。

F02 预登记**所有200条轨迹全部33个观测端点**半步复算；逐坐标保留误差，判据为 abs(q_full-q_half)≤1e-10+1e-8*abs(q_half)，沿 A03 既有 float64 组合容差。包括初点的零误差也保留；不得只报最大绝对误差或放宽容差。若未通过先定位并修正参考方案，再形成可审查记录。

## 6. 数据划分与可恢复采样

200条轨迹按完整轨迹划分：train ID 0000—0159（160），validation 0160—0179（20），test 0180—0199（20）。同轨迹的任何帧、窗口、控制或空间视图不可跨划分，不能先切窗口再随机划分。

初态 |x0| 均匀于 [0.05,1.5]，偶数 ID 取正、奇数取负；y0 均匀于 [-1.5,1.5]，每个划分正负各半。动作与状态和未来独立：每段持续1、4、8 tick等概率选取；20%概率零向量，否则两通道独立均匀于[-0.5,0.5]。末段只截到tick32；不按未来误差、目标或模型输出筛选动作。

数据根种子1703，NumPy PCG64 使用 SeedSequence([1703,trajectory_id,stream_id])，stream 0/1/2 分别管初态/控制/预处理。初态抽样顺序为 |x| 后 y；每段动作先抽时长，再抽零动作事件，仅非零时抽两通道幅值。预处理种子用该 SeedSequence.generate_state(1,dtype=uint64) 转 Python int 交 torch CPU generator。实际 seed/state、库版本随 F02 清单记录。独立轨迹重新生成不得改变其他轨迹，不依赖 Python hash、文件顺序或全局随机消耗。

窗口采样独立 torch CPU generator，seed2703：先均匀抽整数起点，再均匀抽训练总时长；所有对照共用真实样本/查询序列。模型初始化 seed1703 与采样器分离；恢复保存实际 batch cursor 和独立 generator state，而非只重设 seed。

## 7. 训练查询、留出与时间端点

所有条件可用整数起点 j=0…20，保证最长12周期目标不越过tick32。

| 用途 | 控制周期 | 物理秒 |
|---|---|---|
| JEPA监督、时间长支、两个空间支 | {1,2,4,8} | {0.25,0.5,1,2} |
| 时间第一/第二短支，各为 T/2 | {0.5,1,2,4} | {0.125,0.25,0.5,1} |
| 全部训练 Phi 查询 | {0.5,1,2,4,8} | {0.125,0.25,0.5,1,2} |
| 留出总时长 | {3,12} | {0.75,3} |
| 留出 TCD 各短支 | {1.5,6} | {0.375,1.5} |
| 全部留出 Phi 查询 | {1.5,3,6,12} | {0.375,0.75,1.5,3} |

两种总时长的等分都满足 tau=sigma=T/2。训练集合和完全留出集合不交。0.5周期无 JEPA 标签监督，却已用于一致性查询；双零权重条件仍运行这些前向，分别披露“已查询”和“梯度权重为零”。内部积分最大子步0.25周期单列，不冒称独立完整 Phi 查询；“未见”限定相对查询时长，不表示绝对时刻、近似状态或积分子时刻没出现。

遵守 E06 `TrainingBatch`：CPU float64[B] 的 duration 必须逐位等于原始 tau+sigma，observed_time_seconds 与 controls.source_origin_seconds 相同；target_time_seconds 必须逐位等于 observed_time_seconds+duration*p 且严格较晚。选定二进制精确时间格，不借容差修补物理区间。半周期一致性用预测中间 Z 和已知动作路径，不需要真实半周期图像。

## 8. 网格和已验收模型

规范网格16×16，训练一致性目标8×8，留出12×12，均来自同一 raw Z。16→8 用已验收整除平均池化；16→12 用双线性、align_corners=False、edge延拓。目标坐标必须同一 R 作用 source.coords，不能另造均匀坐标冒充已有 R。8×8 已参与训练，不称未见；12×12 是 latent 中间网格插值留出，不称原图分辨率外推，不重新编码其他图像尺寸。

保持 `ProjectionConfig(1408,64)`；GraphConfig 的 q_dim=16、q_seed=1703、sigma_x=0.25、sigma_z=1、max_modes=128，实际 K=min(128,N)，diagnostic_atol=1e-10、rtol=1e-8。Q/带宽固定，整个 eig（特征值和特征向量）停止梯度；X→模态投影等可训练路线不额外 detach。

DynamicsConfig：action_dim=2、dynamic_dim=64、condition_width=64、query_dim=32、top_r=16、spectral_bias_width=32、spatial_bias_width=32、action_hidden_width=64、initial_damping=0.1/周期、head_weight_std=0.001。A/B/N、内容分数、谱偏置、空间偏置及 M_a 全开启。投影、条件网络、空间描述子、初始化和合法 top-r 规则逐项沿 `implementation_decisions.md` 的 A03-D01—D13、A06 对应接口，不引入新网络。

IntegratorConfig：method=exponential_euler、max_step=0.25周期、endpoint_rtol=1e-12、phi1_threshold=1e-4、phi1_order=6。谱计算CPU float64，模型float32，一致性累加float64。早期候选r=8/谱FP32及 smoke 的小K不覆盖已验收设置。

## 9. 损失、对照与未来运行预算

LossConfig：jepa_exponent=1、layer_norm_epsilon=1e-5、time_epsilon=space_epsilon=1e-8，完整模型 time_weight=space_weight=0.001。JEPA 距离沿已验收 target LayerNorm/p=1 mean；时间/空间项严格保留原稿的 Frobenius **平方和**与分母，不转 RMS 或按元素数除，不新增正则。

训练条件只有：参数匹配普通Transformer、完整PO-JEPA双一致性权重均0、完整PO-JEPA双一致性开启；另报未训练 identity。参数匹配架构和相对误差≤1%由F05实际核验，不加无用参数凑数。对照同样可见控制路径、时长、网格和单图raw Z，没有额外历史或隐藏q。本最小对照能评价双一致性的联合收益及相对匹配Transformer的整体表现，不能分别归因时间/空间项或证明图谱优于Fourier。

沿E06 AdamW：lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，无scheduler、clip、AMP、scaler、EMA。拟定每条件2000更新、microbatch=4、无梯度累积；相同有效batch、样本/查询序列和预算。**这是未来F04预算，不解除当前smoke限制。**

F03先在完整4090 24GB或非MIG完整A100（记录40/80GB）验证真实特征、五Phi、三损失联合反向、峰值显存、100更新耗时及验证成本。若资源不足，先报告并在正式比较前版本化地统一降低各条件microbatch；不能暗改K、网格、精度、差别预算，不能假装未实现的梯度累积已可用。F03还须报告三损失原项/加权项、raw预测/目标尺度及有效梯度；不能用一致性变小掩盖尺度退化或静默调权重。

每200更新验证，含step0。只用validation轨迹、起点{0,5,10,15,20}、训练内T={1,2,4,8}、16×16的 future_jepa_distance mean选最小checkpoint，并列取更早step；step0最好则明确无训练收益。test时长、网格、轨迹和Delta不得选checkpoint。首轮模型seed1703仅排错和初判；稳定改进结论再顺序复跑1704/1705，不要求三GPU并行，单种子不能支持稳健性结论。

## 10. 评价、单位和动作诊断

test起点固定{0,5,10,15,20}。ID评价T={1,2,4,8}、网格16×16及8×8；时间留出T={3,12}、16×16；网格留出12×12、T={1,2,4,8}；联合留出12×12、T={3,12}。未来目标在目标网格上为 R(raw_future_target)，预测输入只为 R(raw_start)，真未来不进入预测。

E07主指标名为 future_jepa_distance、tcd_frobenius、scd_frobenius、delta_act；tcd_rms/scd_rms 仅另名附加，不替换Frobenius或训练loss。分位(0.5,0.9,0.95,0.99)，linear插值。保留逐样本、逐轨迹、失败和有效计数；窗口加权 window_weighted_mean 与等轨迹 trajectory_macro_mean 分名，重叠窗口不能当独立轨迹作显著性推断。

TCD=原始Frobenius缺陷/(tau*sigma)，SCD=原始Frobenius缺陷/tau，不加入训练epsilon。p=0.25秒：TCD秒制=TCD周期制/p²=乘16；SCD秒制=SCD周期制/p=乘4，另列单位，不改原指标。

Delta_act=d(替代控制预测,未来目标)-d(执行控制预测,未来目标)，保留正负。在test划分内按轨迹ID循环取下一轨迹（0199回到0180）作donor，同起点j、同时长截已记录控制，记录donor来源和单位。不按未来标签/Delta大小选donor；相同控制导致零Delta仍保留计数，不重抽到正。只作诊断，不加训练；正值不证明完整因果或物理正确。

## 11. 必须保留的未运行门槛

F02仍未运行：参考半步收敛、实际裁剪映射一致且亮斑可见、渲染量化/冻结编码的状态动作信号、最终缓存生成函数的未来扰动检查、实际Z/目标配对/划分/来源。先用**训练划分**预登记诊断对，比较同一当前状态不同合法控制的未来图像/latent，以及不同可见位置；不拿test调渲染或选encoder。

若表示变化只到数值重复水平、identity几乎覆盖全部差异，或同一可见输入对应显著不同未来，暂停效果结论并记录任务/表示瓶颈；不能加入隐藏q/速度/未来图、解冻encoder或新记忆模块越过门槛。有限探针不证明全局Markov充分。F03目标单卡完整训练步、显存、时间与预算仍未运行，3060开发检查不能替代。

形状估算：200×33=6600帧，规范float32 Z共9,515,827,200字节（约9.52GB十进制/8.86GiB），另需RGB、元数据、checkpoint空间；这不是实际磁盘/显存测量。F02按轨迹/小块处理，先冻结编码再释放encoder，后训练预测器；不把全部特征、encoder和所有对照同时驻GPU。不自动下载权重或数据，缺文件/设备暂停对应依赖。

配置的 artifacts.*_sha256/source_files_sha256 目前null，由F02实际产物填写；实际checkpoint路径也未核验。配置自身与协议的最终SHA写**独立验收记录**，不自引用。实际train_step必须具备真实来源全部SHA、train split及实际源码匹配；本模板不能充当来源证明，validation/test不绕过train_step的train划分约束。

## 12. F01验收与后续边界

静态验收检查YAML无重复键、数值类型、配置与本文一致、划分/目标索引、二进制精确端点、查询集合、R方法、已验收参数及来源范围；独立复验重新核算一阶状态理由、四边界、时钟/控制单位、RK4切换、指标换算和未运行门槛。保留真实命令、完整stdout/stderr和退出码。静态检查不是pytest，不伪造XML或测试计数；没有源码行为变更，不重跑2035项。

E07的2020+15是两次输出联合唯一覆盖，不是单次全绿；旧超时/FAILED原件保留。通用辅助器兼容问题仍为已隔离、未修辅助器，沿已核验原生UTF-8替代路线。本步不删警告或改旧证据。

验收、未参与实现者的独立规格/质量复验、本地合main及实际main解析/来源/范围检查全部通过后，F01只登记“协议锁定完成”。此后取得F02 prompt；不能把未来门槛改成通过，也不能把协议完成称为科研复现完成。发现新错误、重大公式/方法歧义或关键输入缺失时，保留证据并暂停对应推进。
