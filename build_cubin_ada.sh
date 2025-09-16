#!/usr/bin/env bash
set -euo pipefail

# Build cubin/fatbin for one or more architectures.
# Override with:
#   - ARCH=<sm_XX or compute_XX>   (single arch)
#   - ARCHES="sm_89,sm_120"        (multi-arch)
#   - RREG=<max registers>
#   - OUT=sha256_kernel.cubin

ARCH=${ARCH:-}
ARCHES=${ARCHES:-}
RREG=${RREG:-}
OUT=${OUT:-sha256_kernel.cubin}

to_sm() {
  local s="$1"
  if [[ "$s" == sm_* ]]; then echo "$s"; return; fi
  if [[ "$s" == compute_* ]]; then echo "${s/compute_/sm_}"; return; fi
  # Convert 8.9 -> sm_89, 12.0 -> sm_120
  s="${s//./}"
  echo "sm_${s}"
}

to_compute() {
  local s="$1"
  if [[ "$s" == compute_* ]]; then echo "$s"; return; fi
  if [[ "$s" == sm_* ]]; then echo "${s/sm_/compute_}"; return; fi
  s="${s//./}"
  echo "compute_${s}"
}

# Resolve architecture list
declare -a ARCH_LIST=()
if [[ -n "$ARCHES" ]]; then
  IFS=',' read -r -a tmp <<< "$ARCHES"
  ARCH_LIST=("${tmp[@]}")
elif [[ -n "$ARCH" ]]; then
  ARCH_LIST=("$ARCH")
else
  # Auto-detect all GPUs' compute capability
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[build] 未找到 nvidia-smi，无法自动检测。请指定 ARCH=sm_89 或 ARCHES=sm_89,sm_120" >&2
    exit 1
  fi
  # Normalize compute capability to MAJ*10+MIN (take only first digit of minor, e.g., 12.2 -> 122, 12.12 -> 121)
  mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d '[:space:]' \
    | sed -E 's/\./ /g' \
    | awk '{min=$2; if(min=="") min=0; if(min!="") { mstr=min; mind=substr(mstr,1,1); if(mind=="") mind=0; } else { mind=0 } printf("%d%d\n", $1, mind)}' \
    | sed -E 's/^0+//; s/^$/0/')
  if [[ ${#caps[@]} -eq 0 ]]; then
    echo "[build] 无法通过 nvidia-smi 获取计算能力。请手动提供 ARCH=sm_89 或 ARCHES=sm_89,sm_120" >&2
    exit 1
  fi
  # Unique and map to sm_XX strings
  mapfile -t uniq_caps < <(printf "%s\n" "${caps[@]}" | sed '/^$/d' | sort -u)
  for c in "${uniq_caps[@]}"; do
    ARCH_LIST+=("sm_${c}")
  done
fi

if [[ ${#ARCH_LIST[@]} -eq 0 ]]; then
  echo "[build] 未得到有效架构列表。" >&2
  exit 1
fi

# Compose NVCC flags
NVCC_FLAGS=( -O3 -Xptxas -O3,-v -Xptxas -dlcm=ca )
for a in "${ARCH_LIST[@]}"; do
  SM=$(to_sm "$a")
  COMPUTE=$(to_compute "$a")
  NVCC_FLAGS+=( -gencode "arch=${COMPUTE},code=${SM}" )
done
if [[ -n "$RREG" ]]; then
  NVCC_FLAGS+=( -maxrregcount=${RREG} )
fi

echo "[build] nvcc ${NVCC_FLAGS[*]} -cubin sha256_kernel.cu -o ${OUT}"
if nvcc "${NVCC_FLAGS[@]}" -cubin sha256_kernel.cu -o "${OUT}"; then
  echo "[build] Wrote ${OUT} (architectures: ${ARCH_LIST[*]})"
  # 成功生成 cubin，不再生成 ptx
  exit 0
else
  echo "[build] 生成 CUBIN 失败，可能是 nvcc 不支持某架构；尝试生成 PTX 作为回退。" >&2
  rm -f "${OUT}" || true
fi

# 仅当 CUBIN 失败时，尝试生成 PTX 回退。
# Prefer lowest compute from ARCH_LIST, fallback to a list of common ones。
declare -a PTX_TRY=()
for a in "${ARCH_LIST[@]}"; do
  c=$(to_compute "$a")
  PTX_TRY+=("${c/compute_/}")
done
# add common fallbacks (highest -> lowest)
PTX_TRY+=(120 110 100 90 89 86 80 75 70 61 60 52)

# de-dup while preserving order
declare -A seen
uniq_try=()
for x in "${PTX_TRY[@]}"; do
  if [[ -z "${seen[$x]:-}" ]]; then uniq_try+=("$x"); seen[$x]=1; fi
done

for v in "${uniq_try[@]}"; do
  echo "[build] 尝试生成 PTX: -arch=compute_${v}"
  if nvcc -O3 -ptx -arch=compute_${v} sha256_kernel.cu -o sha256_kernel.ptx >/dev/null 2>&1; then
    echo "[build] Wrote sha256_kernel.ptx (compute_${v})"
    exit 0
  fi
done

echo "[build] 无法生成 CUBIN 或 PTX。请检查 CUDA 版本与架构支持。" >&2
exit 1
