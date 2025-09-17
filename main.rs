use rustacuda::prelude::*;
use rustacuda::memory::*;
use rustacuda::launch;
use rustacuda::context::CacheConfig;
use std::{env, fs, io};
use std::path::Path;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::mem;
use std::time::{Duration, Instant};
use std::thread::sleep;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::sync::Arc;
use std::io::Write;
use std::io::IsTerminal;
use sha2::{Digest, Sha256};

fn sha256d_hex(msg: &[u8]) -> String {
    let first = Sha256::digest(msg);
    let second = Sha256::digest(&first);
    // 手写 hex，避免再引入 hex crate
    const TBL: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for b in second {
        out.push(TBL[(b >> 4) as usize] as char);
        out.push(TBL[(b & 0x0f) as usize] as char);
    }
    out
}

fn build_message(base: &[u8], nonce: u64, binary_nonce: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(base.len() + 20);
    v.extend_from_slice(base);
    if binary_nonce {
        // 注意：这里采用小端。如你的 kernel 用大端，请改成 to_be_bytes()
        v.extend_from_slice(&nonce.to_le_bytes());
    } else {
        v.extend_from_slice(nonce.to_string().as_bytes());
    }
    v
}

fn print_improvement_json(base: &[u8], nonce: u64, best_lz: u32, binary_nonce: bool) {
    use std::io::{self, Write};

    // 1) 先把“进度行”清掉，避免 JSON 接在进度行尾部
    // 在 TTY 下：回到行首 + ANSI 清行（Windows 10+ 及大多数终端支持）
    if io::stdout().is_terminal() {
        print!("\r\x1b[2K");
    }

    // 2) 计算 hash 并打印「单独一行」JSON
    let msg = build_message(base, nonce, binary_nonce);
    let h = sha256d_hex(&msg);
    let challenge = std::str::from_utf8(base).unwrap_or("<non-utf8>");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    println!(
        r#"{{"nonce":{},"hash":"{}","bestHash":"{}","bestNonce":{},"bestLeadingZeros":{},"challenge":"{}","timestamp":{}}}"#,
        nonce, h, h, nonce, best_lz, challenge, ts
    );
    let _ = io::stdout().flush();
}


#[derive(Clone, Debug)]
struct GpuConfig {
    // global control
    start_nonce: u64,
    total_nonce: u64,
    batch_size: u64,
    // kernel/launch params
    threads_per_block: u32,
    blocks: u32,
    binary_nonce: u32,
    persistent: bool,
    chunk_size: u32,
    ilp: u32,
    progress_ms: u64,
    odometer: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct GpuResult { best_lz: u32, best_nonce: u64 }

#[derive(Debug)]
struct SharedState {
    done: std::sync::atomic::AtomicU64,
    best_lz: std::sync::atomic::AtomicU32,
    best_nonce: std::sync::atomic::AtomicU64,
    finished: std::sync::atomic::AtomicBool,
}

fn run_on_device(device_idx: u32, fixed: &[u8], cfg: &GpuConfig, shared: Option<Arc<SharedState>>) -> Result<GpuResult, Box<dyn Error>> {
    // Each thread creates its own CUDA context and resources bound to one device.
    let device = Device::get_device(device_idx)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Prefer loading cubin (no JIT), fallback to PTX string if loading fails
    let module = if Path::new("sha256_kernel.cubin").exists() {
        match Module::load_from_file(&CString::new("sha256_kernel.cubin")?) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[GPU {}] load cubin failed: {}. Falling back to PTX.", device_idx, e);
                if Path::new("sha256_kernel.ptx").exists() {
                    let ptx = fs::read_to_string("sha256_kernel.ptx")?;
                    Module::load_from_string(&CString::new(ptx)?)?
                } else {
                    return Err(Box::new(io::Error::new(io::ErrorKind::NotFound, "kernel module not found: run ./build_cubin_ada.sh to build sha256_kernel.cubin or sha256_kernel.ptx")));
                }
            }
        }
    } else {
        if Path::new("sha256_kernel.ptx").exists() {
            let ptx = fs::read_to_string("sha256_kernel.ptx")?;
            Module::load_from_string(&CString::new(ptx)?)?
        } else {
            return Err(Box::new(io::Error::new(io::ErrorKind::NotFound, "kernel module not found: run ./build_cubin_ada.sh to build sha256_kernel.cubin or sha256_kernel.ptx")));
        }
    };
    let mut func = module.get_function(CStr::from_bytes_with_nul(b"double_sha256_max_kernel\0")?)?;
    let mut persistent_func = if cfg.binary_nonce == 0 {
        module.get_function(CStr::from_bytes_with_nul(b"double_sha256_persistent_kernel_ascii\0")?)?
    } else {
        module.get_function(CStr::from_bytes_with_nul(b"double_sha256_persistent_kernel\0")?)?
    };
    let reduce_func = module.get_function(CStr::from_bytes_with_nul(b"reduce_best_kernel\0")?)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let progress_stream = if cfg.progress_ms > 0 { Some(Stream::new(StreamFlags::NON_BLOCKING, None)?) } else { None };

    // Hint cache config for L1 (often helps ASCII path)
    let _ = func.set_cache_config(CacheConfig::PreferL1);
    let _ = persistent_func.set_cache_config(CacheConfig::PreferL1);

    let mut d_base = DeviceBuffer::from_slice(fixed)?;

    // Reuse device buffers across batches
    let mut d_block_lz = unsafe { DeviceBuffer::<u32>::zeroed(cfg.blocks as usize)? };
    let mut d_block_nonce = unsafe { DeviceBuffer::<u64>::zeroed(cfg.blocks as usize)? };
    let mut d_best_lz = DeviceBox::new(&0u32)?;
    let mut d_best_nonce = DeviceBox::new(&0u64)?;
    // Live best for persistent progress
    let mut d_best_lz_live = DeviceBox::new(&0u32)?;
    let mut d_best_nonce_live = DeviceBox::new(&0u64)?;
    // Device stop flag for cooperative cancellation
    let mut d_stop_flag = DeviceBox::new(&0u32)?;
    let mut d_next_index = DeviceBox::new(&0u64)?; // for persistent mode

    let mut best_lz = 0u32;
    let mut best_nonce = 0u64;

    let base_len = fixed.len();

    let num_batches = if cfg.persistent { 1 } else if cfg.batch_size == 0 { 1 } else { (cfg.total_nonce + cfg.batch_size - 1) / cfg.batch_size };

    unsafe {
        if cfg.persistent {
            // Reset counters and flags
            d_next_index.copy_from(&0u64)?;
            d_best_lz_live.copy_from(&0u32)?;
            d_best_nonce_live.copy_from(&0u64)?;
            d_stop_flag.copy_from(&0u32)?;

            let warps = (cfg.threads_per_block + 31) / 32;
            let base_rem = (base_len as u32) % 64;
            let shmem_bytes = (warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32)
                + (8 * mem::size_of::<u32>() as u32) + base_rem;
            let enable_live: u32 = if cfg.progress_ms > 0 { 1 } else { 0 };
            launch!(persistent_func<<<cfg.blocks, cfg.threads_per_block, shmem_bytes, stream>>>(
                d_base.as_device_ptr(),
                base_len,
                cfg.start_nonce,
                cfg.total_nonce,
                cfg.binary_nonce,
                d_next_index.as_device_ptr(),
                cfg.chunk_size,
                cfg.ilp,
                enable_live,
                cfg.odometer,
                d_block_nonce.as_device_ptr(),
                d_block_lz.as_device_ptr(),
                d_best_lz_live.as_device_ptr(),
                d_best_nonce_live.as_device_ptr(),
                d_stop_flag.as_device_ptr()
            ))?;

            // Optional progress monitor loop (non-blocking to compute stream)
            if let Some(_prog_stream) = &progress_stream {
                let mut last = 0u64;
                let mut last_printed_lz: u32 = 0; // ← 新增：记录已经打印过的 best_lz

                loop {
                    sleep(Duration::from_millis(cfg.progress_ms));
                    if STOP.load(Ordering::SeqCst) {
                        d_stop_flag.copy_from(&1u32)?;
                        break;
                    }
                    let mut done: u64 = 0;
                    d_next_index.copy_to(&mut done)?;
                            // 读取 live 最优
                    let mut live_lz: u32 = 0;
                    let mut live_nonce: u64 = 0;
                    d_best_lz_live.copy_to(&mut live_lz)?;
                    d_best_nonce_live.copy_to(&mut live_nonce)?;
                            // 只要更优（leading zeros 变大），立即打印一行 JSON
                    if live_lz > last_printed_lz {
                        print_improvement_json(
                            fixed,
                            live_nonce,
                            live_lz,
                            cfg.binary_nonce != 0
                        );
                        last_printed_lz = live_lz;
                    }
                    if let Some(st) = &shared {
                        if done != last {
                            let mut live_lz: u32 = 0;
                            let mut live_nonce: u64 = 0;
                            d_best_lz_live.copy_to(&mut live_lz)?;
                            d_best_nonce_live.copy_to(&mut live_nonce)?;
                            st.done.store(done, Ordering::Relaxed);
                            st.best_lz.store(live_lz, Ordering::Relaxed);
                            st.best_nonce.store(live_nonce, Ordering::Relaxed);
                            last = done;
                        }
                    }
                    if done >= cfg.total_nonce { break; }
                }
            }

            // Ensure persistent kernel completed
            stream.synchronize()?;

            // Reduce on device to a single pair, then readback
            let reduce_threads = {
                let mut t = 1u32;
                while t < cfg.blocks && t < 1024 { t <<= 1; }
                if t > 1024 { 1024 } else { t }
            };
            let reduce_warps = (reduce_threads + 31) / 32;
            let reduce_shmem = reduce_warps * (mem::size_of::<u32>()+mem::size_of::<u64>()) as u32;
            launch!(reduce_func<<<1, reduce_threads, reduce_shmem, stream>>>(
                d_block_lz.as_device_ptr(),
                d_block_nonce.as_device_ptr(),
                cfg.blocks,
                d_best_lz.as_device_ptr(),
                d_best_nonce.as_device_ptr()
            ))?;

            stream.synchronize()?;

            d_best_lz.copy_to(&mut best_lz)?;
            d_best_nonce.copy_to(&mut best_nonce)?;
        } else {
            for batch_idx in 0..num_batches {
                let (start_rel, batch_nonce) = if cfg.batch_size == 0 {
                    (0u64, cfg.total_nonce)
                } else {
                    let start = batch_idx * cfg.batch_size;
                    let bn = std::cmp::min(cfg.batch_size, cfg.total_nonce - start);
                    (start, bn)
                };
                let start_nonce = cfg.start_nonce + start_rel;

                let warps = (cfg.threads_per_block + 31) / 32;
                let base_rem = (base_len as u32) % 64;
                let shmem_bytes = (warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32)
                    + (8 * mem::size_of::<u32>() as u32) + base_rem;
                launch!(func<<<cfg.blocks, cfg.threads_per_block, shmem_bytes, stream>>>(
                    d_base.as_device_ptr(),
                    base_len,
                    start_nonce,
                    batch_nonce,
                    cfg.binary_nonce,
                    d_block_nonce.as_device_ptr(),
                    d_block_lz.as_device_ptr()
                ))?;

                // Reduce on device to a single pair
                let reduce_threads = {
                    // next power of two not exceeding 1024
                    let mut t = 1u32;
                    while t < cfg.blocks && t < 1024 { t <<= 1; }
                    if t > 1024 { 1024 } else { t }
                };
                let reduce_warps = (reduce_threads + 31) / 32;
                let reduce_shmem = reduce_warps * (mem::size_of::<u32>()+mem::size_of::<u64>()) as u32;
                launch!(reduce_func<<<1, reduce_threads, reduce_shmem, stream>>>(
                    d_block_lz.as_device_ptr(),
                    d_block_nonce.as_device_ptr(),
                    cfg.blocks,
                    d_best_lz.as_device_ptr(),
                    d_best_nonce.as_device_ptr()
                ))?;

                stream.synchronize()?;

                let mut batch_lz: u32 = 0;
                let mut batch_nonce_best: u64 = 0;
                d_best_lz.copy_to(&mut batch_lz)?;
                d_best_nonce.copy_to(&mut batch_nonce_best)?;
                if batch_lz > best_lz {
                    best_lz = batch_lz;
                    best_nonce = batch_nonce_best;
                        // ← 新增：发现更优解时打印
                    print_improvement_json(
                        fixed,
                        best_nonce,
                        best_lz,
                        cfg.binary_nonce != 0
                    );
                }

                if batch_idx % 10 == 0 || batch_idx + 1 == num_batches {
                    println!("[GPU {}] Batch {} done, current best_lz={} nonce={} current={}", device_idx, batch_idx, best_lz, best_nonce, start_nonce);
                }
            }
        }
    }

    if let Some(st) = &shared {
        st.finished.store(true, Ordering::SeqCst);
        // publish final best in case it beats the live
        let _ = st.best_lz.fetch_max(best_lz, Ordering::Relaxed);
        // if best_lz improved, also publish nonce
        if st.best_lz.load(Ordering::Relaxed) == best_lz {
            st.best_nonce.store(best_nonce, Ordering::Relaxed);
        }
    }
    Ok(GpuResult { best_lz, best_nonce })
}

static STOP: AtomicBool = AtomicBool::new(false);
// Batch 4300 done, current best_lz=41 nonce=1928769793666 current=4300000000000

fn parse_env_u64(key: &str, default: u64) -> u64 {
    if let Ok(raw) = env::var(key) {
        // Fast path: plain integer
        if let Ok(v) = raw.trim().parse::<u64>() { return v; }
        // Support scientific notation (e.g., 2e7) and suffixes k/m/g/t
        let s = raw.trim().to_ascii_lowercase();
        let mut mult: f64 = 1.0;
        let base = if let Some(last) = s.chars().last() {
            match last {
                'k' => { mult = 1e3; &s[..s.len()-1] }
                'm' => { mult = 1e6; &s[..s.len()-1] }
                'g' => { mult = 1e9; &s[..s.len()-1] }
                't' => { mult = 1e12; &s[..s.len()-1] }
                _ => &s
            }
        } else { &s };
        if let Ok(f) = base.parse::<f64>() {
            if f.is_finite() && f >= 0.0 {
                return (f * mult).floor() as u64;
            }
        }
    }
    default
}

fn parse_env_u32(key: &str, default: u32) -> u32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .map(|v| match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" => true,
            "0" | "false" | "no" | "n" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Ctrl+C -> set stop flag (do not call CUDA in handler)
    let _ = ctrlc::set_handler(|| {
        STOP.store(true, Ordering::SeqCst);
    });
    // 输入：优先取第一个命令行参数，否则走交互输入
    let mut args = env::args().skip(1);
    let base_str = if let Some(s) = args.next() {
        s
    } else {
        println!("输入 txid:index 格式字符串以开始碰撞（例如 c9c4e58e3983558b6fd470b6cb6b444864cb27a878d2db4a3a1ba2538affe22e:2）：");
        let mut s = String::new();
        io::stdin().read_line(&mut s).unwrap();
        s.trim_end().to_string()
    };

    let base_len = base_str.as_bytes().len();
    // 留出 nonce 空间：十进制最长 20 字符；二进制 8 字节
    let reserve = if parse_env_bool("BINARY_NONCE", false) { 8 } else { 20 };
    if base_len + reserve > 128 {
        eprintln!("输入过长：base_len={}，需要 base_len + {} <= 128", base_len, reserve);
        return Ok(());
    }

    // 参数：支持环境变量覆盖，保守默认
    let total_nonce_all: u64 = parse_env_u64("TOTAL_NONCE", 100_000_000_000_000);
    let start_nonce_all: u64 = parse_env_u64("START_NONCE", 0);
    let batch_size: u64 = parse_env_u64("BATCH_SIZE", 1_000_000_000);
    let threads_per_block = parse_env_u32("THREADS", 256);
    let blocks = parse_env_u32("BLOCKS", 1024);
    let binary_nonce = parse_env_bool("BINARY_NONCE", false) as u32;
    let persistent = parse_env_bool("PERSISTENT", false);
    let chunk_size = parse_env_u32("CHUNK", 65536);
    let ilp = parse_env_u32("ILP", 1);
    let progress_ms = parse_env_u64("PROGRESS_MS", 0);
    let odometer = parse_env_bool("ODOMETER", true) as u32;

    rustacuda::init(CudaFlags::empty())?;

    // GPU list: GPU_IDS="0,1,2" or use all available
    let gpu_ids_env = env::var("GPU_IDS").ok();
    let mut gpu_indices: Vec<u32> = if let Some(s) = gpu_ids_env {
        s.split(',')
            .filter_map(|x| x.trim().parse::<u32>().ok())
            .collect()
    } else {
        let n = Device::num_devices()? as u32;
        (0..n).collect()
    };
    gpu_indices.sort_unstable();
    gpu_indices.dedup();
    if gpu_indices.is_empty() { gpu_indices.push(0); }

    // Partition total nonce range into contiguous chunks per GPU
    // Optional weighted split via GPU_WEIGHTS (comma-separated floats aligned to GPU_IDS order).
    let weights_env = env::var("GPU_WEIGHTS").ok();
    let weights: Vec<f64> = if let Some(ws) = weights_env {
        let v: Vec<f64> = ws
            .split(',')
            .filter_map(|x| x.trim().parse::<f64>().ok())
            .collect();
        if v.len() != gpu_indices.len() {
            eprintln!("[warn] GPU_WEIGHTS 数量({}) 与 GPU 数({}) 不一致，忽略，改用等分。", v.len(), gpu_indices.len());
            vec![1.0; gpu_indices.len()]
        } else {
            v
        }
    } else {
        vec![1.0; gpu_indices.len()]
    };
    let sum_w: f64 = weights.iter().copied().sum::<f64>().max(1e-9);
    let mut tasks = Vec::new();
    let mut acc = 0.0f64;
    for (i, &gpu) in gpu_indices.iter().enumerate() {
        let w = weights[i].max(0.0);
        let start_off = ((total_nonce_all as f64) * acc / sum_w).floor() as u64;
        acc += w;
        let end_off = ((total_nonce_all as f64) * acc / sum_w).floor() as u64;
        let len = end_off.saturating_sub(start_off);
        let start = start_nonce_all + start_off;
        if len == 0 { continue; }
        let cfg = GpuConfig {
            start_nonce: start,
            total_nonce: len,
            batch_size,
            threads_per_block,
            blocks,
            binary_nonce,
            persistent,
            chunk_size,
            ilp,
            progress_ms,
            odometer,
        };
        tasks.push((gpu, cfg));
    }

    println!("Launching {} GPU worker(s): {:?}", tasks.len(), gpu_indices);
    for &gpu in &gpu_indices {
        let dev = Device::get_device(gpu)?;
        let name = dev.name()?;
        println!("  - GPU {}: {}", gpu, name);
    }

    let t0 = Instant::now();
    let fixed_bytes: Vec<u8> = base_str.into_bytes();

    // Spawn one thread per GPU
    let mut handles = Vec::new();
    let mut shared_states: Vec<Arc<SharedState>> = Vec::new();
    let mut gpu_t0: Vec<Instant> = Vec::new();
    for (gpu, cfg) in tasks {
        let bytes = fixed_bytes.clone();
        let st = Arc::new(SharedState{
            done: std::sync::atomic::AtomicU64::new(0),
            best_lz: std::sync::atomic::AtomicU32::new(0),
            best_nonce: std::sync::atomic::AtomicU64::new(0),
            finished: std::sync::atomic::AtomicBool::new(false),
        });
        shared_states.push(st.clone());
        gpu_t0.push(Instant::now());
        let st_for_err = st.clone();
        handles.push(thread::spawn(move || -> Result<(u32, u64), Box<dyn Error + Send + Sync>> {
            match run_on_device(gpu, &bytes, &cfg, Some(st)) {
                Ok(res) => Ok((res.best_lz, res.best_nonce)),
                Err(e) => {
                    st_for_err.finished.store(true, Ordering::SeqCst);
                    Err(Box::new(io::Error::new(io::ErrorKind::Other, e.to_string())))
                }
            }
        }));
    }

    // Aggregated progress printer (single line) when progress_ms > 0
    if progress_ms > 0 {
        let total_all: u64 = total_nonce_all;
        let inline = std::io::stdout().is_terminal();
        loop {
            thread::sleep(Duration::from_millis(progress_ms));
            // compute per-GPU rates and global progress
            let mut sum_done = 0u64;
            let mut rates: Vec<f64> = Vec::with_capacity(shared_states.len());
            for (i, st) in shared_states.iter().enumerate() {
                let done = st.done.load(Ordering::Relaxed);
                sum_done = sum_done.saturating_add(done);
                let secs = gpu_t0[i].elapsed().as_secs_f64();
                let r = if secs > 0.0 { (done as f64) / secs / 1e9 } else { 0.0 };
                rates.push(r);
            }
            let pct = (sum_done as f64) / (total_all as f64) * 100.0;
            // global best from live metrics
            let mut g_best_lz = 0u32;
            let mut g_best_nonce = 0u64;
            for st in &shared_states {
                let lz = st.best_lz.load(Ordering::Relaxed);
                let no = st.best_nonce.load(Ordering::Relaxed);
                if lz > g_best_lz { g_best_lz = lz; g_best_nonce = no; }
            }
            // compose single line
            let mut line = format!("Progress: {:.2}% |", pct.min(100.0));
            for (i, r) in rates.iter().enumerate() {
                if i > 0 { line.push(' '); }
                line.push_str(&format!(" GPU{} {:.2} GH/s", gpu_indices[i], r));
                if i + 1 < rates.len() { line.push(','); }
            }
            line.push_str(&format!(" | best_lz={} nonce={}", g_best_lz, g_best_nonce));
            if inline {
                print!("\r{}", line);
                let _ = std::io::stdout().flush();
            } else {
                println!("{}", line);
            }

            // stop when all finished
            if shared_states.iter().all(|s| s.finished.load(Ordering::SeqCst)) { break; }
            // no-op
        }
        if inline { println!(""); }
    }

    // Gather results
    let mut best_lz = 0u32;
    let mut best_nonce = 0u64;
    for (idx, h) in handles.into_iter().enumerate() {
        match h.join() {
            Ok(Ok((lz, nonce))) => {
                println!("[GPU {}] done: best_lz={} nonce={}", gpu_indices[idx], lz, nonce);
                if lz > best_lz { best_lz = lz; best_nonce = nonce; }
            }
            Ok(Err(e)) => {
                eprintln!("[GPU {}] error: {}", gpu_indices[idx], e);
            }
            Err(_) => {
                eprintln!("[GPU {}] thread panicked", gpu_indices[idx]);
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let ghps = if elapsed > 0.0 { (total_nonce_all as f64) / elapsed / 1e9 } else { 0.0 };
    println!("Final best leading zeros: {} at nonce {}", best_lz, best_nonce);
    println!("Summary: GPUs={} elapsed {:.2}s, rate {:.2} GH/s", gpu_indices.len(), elapsed, ghps);

    Ok(())
}

