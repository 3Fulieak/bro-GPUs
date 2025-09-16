# bro: 多 GPU 双重 SHA‑256 碰撞搜索器（Rust + CUDA）

本项目在 GPU 上并行搜索使 `SHA256(SHA256(base + nonce))` 具有最多前导零位（leading zeros）的 `nonce`。

- 多 GPU 并行：按权重切分工作量，每张卡独立推进进度。
- 两种执行模式：
  - 批处理模式（默认）：主机按批次下发工作，适合中小规模或调试。
  - 持久化内核（`PERSISTENT=1`）：设备端自取任务，减少往返开销，适合长时间跑满。
- `nonce` 形式可选：十进制 ASCII（默认）或 8 字节二进制（`BINARY_NONCE=1`）。
- 内核加载：优先加载 `sha256_kernel.cubin`，失败时回退 `sha256_kernel.ptx`（均由 `nvcc` 生成）。
- 进度与速率：可周期性打印总体进度与各 GPU 实时 GH/s；结束时输出最佳结果与总体吞吐。


**环境要求**
- NVIDIA GPU 与驱动（可用 `nvidia-smi` 检查）。
- CUDA Toolkit（含 `nvcc`，用于构建 CUBIN/PTX）。
- Rust 工具链（`cargo`，edition 2021）。
- 建议 Linux 环境；Windows/WSL 需确保 CUDA 可用。macOS 无原生 NVIDIA 支持。


**快速安装与启动流程**
- 安装 Rust（按提示完成安装）：
  - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- 配置环境变量（当前会话生效，建议写入 shell 配置文件）：
  - `export PATH="$HOME/.cargo/bin:$PATH"`
- 检查是否安装成功：
  - `rustc --version`
  - `cargo --version`
- 赋予构建脚本执行权限（首次或上传后执行一次）：
  - `chmod +x build_cubin_ada.sh`
- 生成 CUBIN（失败时自动回退 PTX，通常只需执行一次）：
  - `./build_cubin_ada.sh`
- 启动（示例参数，持久化内核 + 进度，每秒打印一次）：
  - `PERSISTENT=1 BINARY_NONCE=0 BLOCKS=8192 THREADS=512 ILP=4 CHUNK=1024000 PROGRESS_MS=1000 TOTAL_NONCE=5000000000000 cargo run --release`
- 查看 GPU 占用率：
  - `nvidia-smi`

注：以上“生成 CUBIN/赋权”通常只需运行一次；后续根据需要直接“启动”。


**目录结构**
- `Cargo.toml`：二进制 crate 配置，bin 名称为 `bro`。
- `main.rs`：入口与多 GPU 调度、参数解析、进度与结果汇总。
- `sha256_kernel.cu`：CUDA 内核（`double_sha256_max_kernel`、`double_sha256_persistent_kernel(_ascii)`、`reduce_best_kernel` 等）。
- `build_cubin_ada.sh`：生成多架构 `sha256_kernel.cubin`，失败时回退生成 `sha256_kernel.ptx`。


**构建与运行**
- 生成内核（推荐先生成跨卡 CUBIN）：
  - 自动枚举本机架构并生成：
    - `./build_cubin_ada.sh`
  - 指定单架构：
    - `ARCH=sm_89 ./build_cubin_ada.sh`
  - 指定多架构：
    - `ARCHES="sm_89,sm_120" ./build_cubin_ada.sh`
  - 限制寄存器（可影响性能/占用）：
    - `RREG=80 ./build_cubin_ada.sh`

- 运行（交互输入或传参）：
  - 交互：`cargo run --release`
  - 传参：`cargo run --release -- "txid:index"`

- 持久化模式（单卡示例，1 秒打印一次进度）：
  - `PERSISTENT=1 PROGRESS_MS=1000 TOTAL_NONCE=2e11 cargo run --release -- "55cc38e8bafeae691465e2bb2:0"`

- 多卡（按权重切分工作量）：
  - `GPU_IDS=0,1 GPU_WEIGHTS=1.0,0.9 PERSISTENT=1 THREADS=256 BLOCKS=1024 ILP=2 CHUNK=262144 PROGRESS_MS=1000 TOTAL_NONCE=1e14 cargo run --release -- "c9c4...:2"`


**输入与限制**
- 输入为 `base` 字符串：命令行第一个参数或交互输入（示例格式 `txid:index`）。程序会在其尾部拼接 `nonce` 再进行双 SHA‑256。
- 长度限制：`base_len + (ASCII 时 20 | 二进制时 8) <= 128`，超出将直接返回并提示错误。


**环境变量一览（含默认）**
- 扫描范围：
  - `TOTAL_NONCE`（默认 `1e14`）：总尝试量，支持 `k/m/g/t` 后缀与科学计数法（如 `2e7`）。
  - `START_NONCE`（默认 `0`）：起始 `nonce`。
  - `BATCH_SIZE`（默认 `1e9`）：批处理模式下的单批大小；`PERSISTENT=1` 时忽略；`0` 表示单批全量。
- 内核/调度：
  - `THREADS`（默认 `256`）：每 block 线程数。
  - `BLOCKS`（默认 `1024`）：block 数。
  - `ILP`（默认 `1`）：内核内指令级并行度。
  - `CHUNK`（默认 `65536`）：持久化模式每次抓取的任务块大小。
  - `PERSISTENT`（默认 `0`）：启用持久化内核。
  - `BINARY_NONCE`（默认 `0`）：`1` 为 8 字节二进制 `nonce`，`0` 为十进制 ASCII。
  - `PROGRESS_MS`（默认 `0`）：>0 则按间隔打印全局进度与各 GPU 速率。
  - `ODOMETER`（默认 `1`）：持久化模式的计数/进度开关。
- 多 GPU：
  - `GPU_IDS`：逗号分隔 GPU 索引；缺省使用全部可用设备。
  - `GPU_WEIGHTS`：与 `GPU_IDS` 对齐的浮点权重，用于总量按性能加权切分。


**输出示例与含义**
- 启动：枚举到的 GPU 与名称。
- 周期进度（当 `PROGRESS_MS>0`）：
  - `Progress: 12.34% | GPU0 3.10 GH/s, GPU1 3.05 GH/s | best_lz=41 nonce=1928769793666`
- 每 GPU 完成：
  - `[GPU X] done: best_lz=.. nonce=..`
- 总结：
  - `Final best leading zeros: <lz> at nonce <n>`
  - `Summary: GPUs=<n> elapsed <s>, rate <GH/s>`


**手动性能调优（仓库不含自动调参脚本）**
- 步骤 1：确保生成匹配硬件的 `sha256_kernel.cubin`；若 CUBIN 生成失败会自动回退 PTX（可能略慢）。
  - 建议对所有在用卡生成多架构 CUBIN：`ARCHES="sm_89,sm_120,..." ./build_cubin_ada.sh`
- 步骤 2：逐卡试参，观察 `Summary` 的 `rate` 或进度行中的 GH/s。
  - 隔离单卡：`CUDA_VISIBLE_DEVICES=0 ... cargo run --release -- "base"`
  - 常见搜索范围：
    - `THREADS`：`256~512`
    - `BLOCKS`：`512~1536`
    - `ILP`：`1~4`
    - `CHUNK`（仅持久化）：`65536~524288`
    - `RREG`（通过 `build_cubin_ada.sh` 的 `RREG`）：`64~80`（增大可能降占用）
  - 长时间跑建议：`PERSISTENT=1 PROGRESS_MS=0` 以降低主机-设备同步开销。
- 步骤 3：根据各卡实测 GH/s 设置 `GPU_WEIGHTS`。
  - 例如 GPU0=3.1 GH/s、GPU1=2.9 GH/s，则 `GPU_WEIGHTS=3.1,2.9`。


**常见问题**
- 报错 `kernel module not found`：先运行 `./build_cubin_ada.sh` 生成 `sha256_kernel.cubin` 或 `sha256_kernel.ptx`。
- 找不到 `nvcc`：请安装 CUDA Toolkit 并将 `nvcc` 加入 PATH。
- 没有 `nvidia-smi`：无法自动检测架构，请用 `ARCH/ARCHES` 明确指定目标架构。
- 输入过长：需满足 `base_len + 20(ASCII) 或 8(二进制) <= 128`。
- 性能异常：优先使用 CUBIN；根据上述“手动性能调优”逐项验证 `THREADS/BLOCKS/ILP/CHUNK/RREG`。
- 中断退出：按 Ctrl+C；持久化模式下会触发设备端停止标志，稍候清理完成。


**注意事项**
- 长时间运行会高负载占用 GPU，请关注温度与功耗（可能需要设置功耗/频率上限）。
- 本项目仅供研究与学习用途，请在合法合规前提下使用。
