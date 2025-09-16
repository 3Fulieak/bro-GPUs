#include <stdint.h>
#include <string.h>

// Forward declaration for word-based compressor
__device__ void sha256_compress_words(const uint32_t W0[16], uint32_t H[8]);

// Constant memory for SHA-256 K table
__constant__ uint32_t Kc[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Two-digit zero-padded table for fast itoa ("00".."99")
__device__ __constant__ char DIG2[201] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

__device__ inline uint32_t rotr(uint32_t x, int n) {
    // Use funnel shift for faster rotates on modern SMs
    return __funnelshift_r(x, x, n);
}

__device__ void sha256_compress_block(const uint8_t block[64], uint32_t H[8]) {
    // Rolling W[16] schedule to reduce register/local memory pressure
    uint32_t W[16];
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        W[t] = ((uint32_t)block[t*4+0]<<24)|((uint32_t)block[t*4+1]<<16)|((uint32_t)block[t*4+2]<<8)|((uint32_t)block[t*4+3]);
    }
    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll 64
    for(int t=0;t<64;t++){
        uint32_t wt;
        if(t < 16){
            wt = W[t];
        } else {
            uint32_t w15 = W[(t-15)&15];
            uint32_t w2  = W[(t-2)&15];
            uint32_t s0 = rotr(w15,7)^rotr(w15,18)^(w15>>3);
            uint32_t s1 = rotr(w2,17)^rotr(w2,19)^(w2>>10);
            wt = W[t&15] + s0 + W[(t-7)&15] + s1;
            W[t&15] = wt;
        }
        uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t temp1=h+S1+ch+Kc[t]+wt;
        uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t temp2=S0+maj;
        h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
}

__device__ void sha256_multi_block(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t nblocks = (len+9+63)/64;
    for(size_t blk=0; blk<nblocks; blk++){
        uint8_t block[64] = {0};
        #pragma unroll
        for(int i=0;i<64;i++){
            size_t idx = blk*64 + i;
            if(idx<len) block[i]=msg[idx];
            else if(idx==len) block[i]=0x80;
        }
        if(blk==nblocks-1){
            uint64_t bit_len=len*8;
            #pragma unroll
            for(int i=0;i<8;i++) block[56+i]=(bit_len>>(56-8*i))&0xff;
        }
        sha256_compress_block(block,H);
    }
    for(int i=0;i<8;i++){
        out[i*4+0]=(H[i]>>24)&0xff;
        out[i*4+1]=(H[i]>>16)&0xff;
        out[i*4+2]=(H[i]>>8)&0xff;
        out[i*4+3]=(H[i])&0xff;
    }
}

// Optimized path for hashing exactly 32 bytes
__device__ void sha256_32B(const uint8_t in[32], uint8_t out[32]){
    uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[64] = {0};
    #pragma unroll
    for(int i=0;i<32;i++) block[i]=in[i];
    block[32]=0x80;
    // bit length = 32*8 = 256
    block[63] = 256 & 0xff;
    block[62] = (256>>8) & 0xff;
    // upper bytes already 0
    sha256_compress_block(block, H);
    #pragma unroll
    for(int i=0;i<8;i++){
        out[i*4+0]=(H[i]>>24)&0xff;
        out[i*4+1]=(H[i]>>16)&0xff;
        out[i*4+2]=(H[i]>>8)&0xff;
        out[i*4+3]=(H[i])&0xff;
    }
}

// Optimized: SHA-256 for exactly 32B when provided as 8 big-endian words
__device__ void sha256_32B_from_words(const uint32_t in[8], uint32_t out_words[8]){
    // IV
    uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t W0[16];
    #pragma unroll
    for(int i=0;i<8;i++) W0[i]=in[i];
    W0[8]=0x80000000u;
    #pragma unroll
    for(int i=9;i<14;i++) W0[i]=0u;
    W0[14]=0u; W0[15]=256u; // 32*8 bits
    sha256_compress_words(W0, H);
    #pragma unroll
    for(int i=0;i<8;i++) out_words[i]=H[i];
}

__device__ inline uint32_t count_leading_zeros_words(const uint32_t* w){
    uint32_t cnt=0;
    #pragma unroll
    for(int i=0;i<8;i++){
        uint32_t v = w[i];
        if(v==0u){ cnt+=32; continue; }
        cnt += __clz(v);
        break;
    }
    return cnt;
}

// Compress a single 512-bit block given as 16 big-endian 32-bit words
__device__ void sha256_compress_words(const uint32_t W0[16], uint32_t H[8]){
    uint32_t W[16];
    #pragma unroll
    for(int i=0;i<16;i++) W[i] = W0[i];
    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll 64
    for(int t=0;t<64;t++){
        uint32_t wt;
        if(t<16){ wt = W[t]; }
        else {
            uint32_t w15 = W[(t-15)&15];
            uint32_t w2  = W[(t-2)&15];
            uint32_t s0 = rotr(w15,7)^rotr(w15,18)^(w15>>3);
            uint32_t s1 = rotr(w2,17)^rotr(w2,19)^(w2>>10);
            wt = W[t&15] + s0 + W[(t-7)&15] + s1;
            W[t&15] = wt;
        }
        uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t temp1=h+S1+ch+Kc[t]+wt;
        uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t temp2=S0+maj;
        h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
}

// Fast itoa for u64 using DIG2, returns length, writes forward
__device__ inline int itoa_u64_fast(uint64_t v, char* out){
    char buf[20];
    int idx = 20;
    while(v >= 100ULL){
        uint64_t q = v / 100ULL;
        uint32_t r = (uint32_t)(v - q * 100ULL);
        idx -= 2;
        buf[idx]   = DIG2[r*2+0];
        buf[idx+1] = DIG2[r*2+1];
        v = q;
    }
    if(v < 10ULL){
        idx -= 1;
        buf[idx] = (char)('0' + (char)v);
    } else {
        idx -= 2;
        uint32_t r = (uint32_t)v;
        buf[idx]   = DIG2[r*2+0];
        buf[idx+1] = DIG2[r*2+1];
    }
    int len = 20 - idx;
    #pragma unroll
    for(int i=0;i<len;i++) out[i] = buf[idx+i];
    return len;
}

// Fast itoa using 4-digit groups with DIG2 composition (fewer divisions)
__device__ inline void udivmod10000_dev(uint64_t x, uint64_t* q, uint32_t* r){
    const uint64_t M = 1844674407370956ull; // ceil(2^64/10000)
    uint64_t qh = __umul64hi(x, M);
    uint64_t rem = x - qh * 10000ull;
    if(rem >= 10000ull){ rem -= 10000ull; qh += 1ull; }
    *q = qh; *r = (uint32_t)rem;
}

__device__ inline int itoa_u64_fast4(uint64_t v, char* out){
    char buf[20];
    int idx = 20;
    while(v >= 10000ULL){
        uint64_t q; uint32_t r;
        udivmod10000_dev(v, &q, &r);
        idx -= 4;
        uint32_t hi = r / 100u;
        uint32_t lo = r - hi*100u;
        buf[idx+0] = DIG2[hi*2+0];
        buf[idx+1] = DIG2[hi*2+1];
        buf[idx+2] = DIG2[lo*2+0];
        buf[idx+3] = DIG2[lo*2+1];
        v = q;
    }
    // v < 10000: write minimal digits (no leading zeros)
    if(v >= 1000ULL){
        uint32_t r = (uint32_t)v;
        uint32_t hi = r / 100u;
        uint32_t lo = r % 100u;
        idx -= 4;
        buf[idx+0] = DIG2[hi*2+0];
        buf[idx+1] = DIG2[hi*2+1];
        buf[idx+2] = DIG2[lo*2+0];
        buf[idx+3] = DIG2[lo*2+1];
    } else if(v >= 100ULL){
        uint32_t r = (uint32_t)v;
        uint32_t hi = r / 100u;   // 1..9
        uint32_t lo = r % 100u;   // 00..99
        idx -= 3;
        buf[idx+0] = (char)('0' + (char)hi);
        buf[idx+1] = DIG2[lo*2+0];
        buf[idx+2] = DIG2[lo*2+1];
    } else if(v >= 10ULL){
        uint32_t r = (uint32_t)v;
        uint32_t hi = r / 10u;    // 1..9
        uint32_t lo = r % 10u;    // 0..9
        idx -= 2;
        buf[idx+0] = (char)('0' + (char)hi);
        buf[idx+1] = (char)('0' + (char)lo);
    } else {
        idx -= 1;
        buf[idx] = (char)('0' + (char)v);
    }
    int len = 20 - idx;
    #pragma unroll
    for(int i=0;i<len;i++) out[i] = buf[idx+i];
    return len;
}

__device__ inline uint32_t count_leading_zeros32(const uint8_t* h){
    // Scan by 32-bit big-endian words, use __clz for speed
    uint32_t cnt = 0;
    #pragma unroll
    for(int w=0; w<8; ++w){
        uint32_t v = ((uint32_t)h[w*4+0]<<24)|((uint32_t)h[w*4+1]<<16)|((uint32_t)h[w*4+2]<<8)|((uint32_t)h[w*4+3]);
        if(v==0){ cnt += 32; continue; }
        cnt += __clz(v);
        break;
    }
    return cnt;
}

__device__ inline void warp_reduce_max(uint32_t &lz, uint64_t &nonce){
    unsigned mask = 0xffffffffu;
    for(int offset=16; offset>0; offset>>=1){
        uint32_t lz_o = __shfl_down_sync(mask, lz, offset);
        uint64_t no_o = __shfl_down_sync(mask, nonce, offset);
        if(lz_o > lz){ lz = lz_o; nonce = no_o; }
    }
}

// Kernel: 每 block 找局部最大，返回 block 结果
extern "C" __global__
void double_sha256_max_kernel(const uint8_t* __restrict__ base_message, size_t base_len,
    uint64_t start_nonce, uint64_t total_nonce, uint32_t binary_nonce,
    uint64_t* block_best_nonce, uint32_t* block_best_lz)
{
    extern __shared__ uint8_t sdata[];
    const int warps = (blockDim.x + 31) >> 5;
    // sdata 布局: [warp_lz[warps]][warp_nonce[warps]][midstate 8*u32][base_rem[rem]]
    uint32_t* s_lz = (uint32_t*)sdata;
    uint64_t* s_nonce = (uint64_t*)(s_lz + warps);
    uint32_t* s_mid = (uint32_t*)(s_nonce + warps);
    uint8_t* s_base_rem = (uint8_t*)(s_mid + 8);

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t stride = blockDim.x * gridDim.x;

    uint32_t local_best_lz = 0;
    uint64_t local_best_nonce = 0;

    // 预计算 midstate（base 的整块）并缓存 base 的剩余部分
    const size_t full_blocks = base_len / 64;
    const int rem = (int)(base_len % 64);
    if(threadIdx.x == 0){
        uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        for(size_t b=0; b<full_blocks; ++b){
            const uint8_t* blk = base_message + b*64;
            sha256_compress_block(blk, H);
        }
        #pragma unroll
        for(int i=0;i<8;i++) s_mid[i]=H[i];
        #pragma unroll
        for(int i=0;i<rem;i++) s_base_rem[i] = base_message[full_blocks*64 + i];
    }
    __syncthreads();

    for(uint64_t i = tid; i < total_nonce; i += stride){
        uint64_t nonce = start_nonce + i;

        // 从 midstate 出发，仅压缩尾部（base 剩余 + nonce + padding），最多2块
        uint32_t H1[8];
        #pragma unroll
        for(int i2=0;i2<8;i2++) H1[i2] = s_mid[i2];

        // Compute nonce length without building buffers
        int nonce_len; if(binary_nonce){ nonce_len = 8; } else { char lenbuf[20]; nonce_len = itoa_u64_fast4(nonce, lenbuf); }
        // 计算总比特长度
        uint64_t bit_len = (uint64_t)(base_len + (uint64_t)nonce_len) * 8ull;

        if(rem + nonce_len <= 55){
            // Fast one-block path: assemble 16 words directly
            uint32_t W0[16];
            #pragma unroll
            for(int i4=0;i4<16;i4++) W0[i4]=0u;
            int wpos = 0;
            // base remainder
            #pragma unroll
            for(int j=0;j<rem;j++){
                int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                W0[wi] |= ((uint32_t)s_base_rem[j]) << sh; wpos++;
            }
            if(binary_nonce){
                #pragma unroll
                for(int j=0;j<8;j++){
                    uint8_t by = (uint8_t)((nonce >> (8*j)) & 0xff);
                    int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                    W0[wi] |= ((uint32_t)by) << sh; wpos++;
                }
            } else {
                char ascbuf[20]; int alen = itoa_u64_fast4(nonce, ascbuf);
                for(int j=0;j<alen;j++){ int wi = wpos>>2; int sh = 24-((wpos&3)*8); W0[wi] |= ((uint32_t)ascbuf[j])<<sh; wpos++; }
            }
            // 0x80 pad
            { int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8); W0[wi] |= (0x80u << sh); wpos++; }
            // bit length (big-endian words)
            uint64_t blen = (uint64_t)(base_len + (uint64_t)wpos - (uint64_t)rem - 1ull) * 8ull; // rebuild from written bytes
            W0[14] = (uint32_t)(blen >> 32);
            W0[15] = (uint32_t)(blen & 0xffffffffu);
            sha256_compress_words(W0, H1);
        } else {
            // Two-block path: assemble both blocks in words
            // First block
            uint32_t W0a[16];
            #pragma unroll
            for(int ii=0; ii<16; ++ii) W0a[ii]=0u;
            int wpos = 0;
            // base remainder bytes
            #pragma unroll
            for(int j=0;j<rem;j++){
                int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                W0a[wi] |= ((uint32_t)s_base_rem[j]) << sh; wpos++;
            }
            // nonce bytes split across blocks
            int cap = 64 - rem;
            int n1 = nonce_len < cap ? nonce_len : cap;
            if(binary_nonce){
                #pragma unroll
                for(int j=0;j<n1;j++){
                    uint8_t by = (uint8_t)((nonce >> (8*j)) & 0xff);
                    int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                    W0a[wi] |= ((uint32_t)by) << sh; wpos++;
                }
            } else {
                { char asc1[20]; int a1 = itoa_u64_fast4(nonce, asc1);
                  int up = (a1 < n1 ? a1 : n1);
                  for(int j=0;j<up;j++){ int wi=wpos>>2; int sh=24-((wpos&3)*8); W0a[wi] |= ((uint32_t)asc1[j])<<sh; wpos++; } }
            }
            int wrote80_in_blk1 = 0;
            if(wpos < 64){ int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8); W0a[wi] |= (0x80u << sh); wrote80_in_blk1 = 1; }
            sha256_compress_words(W0a, H1);

            // Second block
            uint32_t W0b[16];
            #pragma unroll
            for(int ii=0; ii<16; ++ii) W0b[ii]=0u;
            int wpos2 = 0;
            int n2 = nonce_len - n1;
            if(binary_nonce){
                #pragma unroll
                for(int j=0;j<n2;j++){
                    uint8_t by = (uint8_t)((nonce >> (8*(j+n1))) & 0xff);
                    int wi = wpos2 >> 2; int sh = 24 - ((wpos2 & 3) * 8);
                    W0b[wi] |= ((uint32_t)by) << sh; wpos2++;
                }
            } else {
                { char asc2[20]; int a2 = itoa_u64_fast4(nonce, asc2);
                  for(int j=0;j<n2 && (n1+j)<a2; j++){ int wi=wpos2>>2; int sh=24-((wpos2&3)*8); W0b[wi] |= ((uint32_t)asc2[n1+j])<<sh; wpos2++; } }
            }
            if(!wrote80_in_blk1){ int wi = wpos2 >> 2; int sh = 24 - ((wpos2 & 3) * 8); W0b[wi] |= (0x80u << sh); wpos2++; }
            // write bit length at words 14..15
            W0b[14] = (uint32_t)(bit_len >> 32);
            W0b[15] = (uint32_t)(bit_len & 0xffffffffu);
            sha256_compress_words(W0b, H1);
        }

        // 第二轮：直接用 H1 作为 8 word 输入
        uint32_t H2[8];
        sha256_32B_from_words(H1, H2);
        uint32_t lz = count_leading_zeros_words(H2);

        if(lz > local_best_lz){
            local_best_lz = lz;
            local_best_nonce = nonce;
        }
    }

    // Warp-level 归约到每个 warp 的 lane0
    warp_reduce_max(local_best_lz, local_best_nonce);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp_id = threadIdx.x >> 5;
    if(lane == 0){
        s_lz[warp_id] = local_best_lz;
        s_nonce[warp_id] = local_best_nonce;
    }
    __syncthreads();

    // 由 warp0 对各 warp 的结果做二次归约
    if(warp_id == 0){
        uint32_t v_lz = (threadIdx.x < (blockDim.x + 31)/32) ? s_lz[lane] : 0u;
        uint64_t v_no = (threadIdx.x < (blockDim.x + 31)/32) ? s_nonce[lane] : 0ull;
        warp_reduce_max(v_lz, v_no);
        if(lane == 0){
            block_best_lz[blockIdx.x] = v_lz;
            block_best_nonce[blockIdx.x] = v_no;
        }
    }
}

// Persistent kernel: fetch work via global atomic counter
extern "C" __global__
void double_sha256_persistent_kernel(const uint8_t* __restrict__ base_message, size_t base_len,
    uint64_t start_nonce, uint64_t total_nonce, uint32_t binary_nonce,
    unsigned long long* next_index, // global atomic counter
    uint32_t chunk_size, uint32_t iters_per_thread, uint32_t enable_live, uint32_t odometer_ascii,
    uint64_t* block_best_nonce, uint32_t* block_best_lz,
    uint32_t* g_best_lz_live, uint64_t* g_best_nonce_live,
    const volatile uint32_t* stop_flag)
{
    extern __shared__ uint8_t sdata[];
    const int warps = (blockDim.x + 31) >> 5;
    uint32_t* s_lz = (uint32_t*)sdata;
    uint64_t* s_nonce = (uint64_t*)(s_lz + warps);
    uint32_t* s_mid = (uint32_t*)(s_nonce + warps);
    uint8_t* s_base_rem = (uint8_t*)(s_mid + 8);

    // midstate + base remainder
    const size_t full_blocks = base_len / 64;
    const int rem = (int)(base_len % 64);
    if(threadIdx.x == 0){
        uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        for(size_t b=0; b<full_blocks; ++b){
            const uint8_t* blk = base_message + b*64;
            sha256_compress_block(blk, H);
        }
        #pragma unroll
        for(int i=0;i<8;i++) s_mid[i]=H[i];
        #pragma unroll
        for(int i=0;i<rem;i++) s_base_rem[i] = base_message[full_blocks*64 + i];
    }
    __syncthreads();

    uint32_t local_best_lz = 0;
    uint64_t local_best_nonce = 0;

    // Persistent, block-chunk work distribution
    __shared__ unsigned long long block_start;
    __shared__ unsigned int s_g_best;
    for(;;){
        if(*stop_flag){ break; }
        if(threadIdx.x == 0){
            block_start = atomicAdd(next_index, (unsigned long long)chunk_size);
            if(enable_live){ s_g_best = *g_best_lz_live; }
        }
        __syncthreads();
        if(block_start >= total_nonce) break;
        unsigned long long block_end = block_start + (unsigned long long)chunk_size;
        if(block_end > total_nonce) block_end = total_nonce;

        // Process chunk in thread-local contiguous segments of length iters_per_thread
        for(unsigned long long seg = block_start + (unsigned long long)threadIdx.x * (unsigned long long)iters_per_thread;
            seg < block_end; seg += (unsigned long long)blockDim.x * (unsigned long long)iters_per_thread){

            // Prepare first nonce in this segment
            unsigned long long idx0 = seg;
            if(idx0 >= block_end) break;
            uint64_t nonce0 = start_nonce + (uint64_t)idx0;

            // Local decimal buffer for ASCII mode
            char decbuf[21]; int dec_len = 0;
            if(!binary_nonce){ dec_len = itoa_u64_fast4(nonce0, decbuf); }

            // Iterate within segment (consecutive +1)
            #pragma unroll 4
            for(uint32_t iter=0; iter<4; ++iter){
                if(iter >= iters_per_thread) break;
                unsigned long long idx = seg + (unsigned long long)iter;
                if(idx >= block_end) break;
                uint64_t nonce = start_nonce + (uint64_t)idx;

                if(!binary_nonce && odometer_ascii && iter>0){
                    // odometer +1 (decimal increment)
                    int i = dec_len - 1;
                    for(; i>=0; --i){
                        if(decbuf[i] != '9'){ decbuf[i] += 1; break; }
                        decbuf[i] = '0';
                    }
                    if(i < 0){ // overflow prepend '1'
                        for(int j=dec_len; j>0; --j) decbuf[j] = decbuf[j-1];
                        decbuf[0] = '1';
                        dec_len += 1;
                    }
                }

                // Tail blocks build
                uint32_t H1[8];
                #pragma unroll
                for(int i2=0;i2<8;i2++) H1[i2] = s_mid[i2];

                // Compute nonce length and prepare pointer to ascii buffer when needed
                const char* usep = NULL; int nonce_len = 0;
                if(binary_nonce){ nonce_len = 8; }
                else {
                    if(odometer_ascii && iter>0){ usep = decbuf; nonce_len = dec_len; }
                    else { char tmpb[20]; int nlen = itoa_u64_fast4(nonce, tmpb); usep = tmpb; nonce_len = nlen; }
                }
                uint64_t bit_len = (uint64_t)(base_len + (uint64_t)nonce_len) * 8ull;

                if(rem + nonce_len <= 55){
                    // One-block fast path with direct word assembly
                    uint32_t W0[16];
                    #pragma unroll
                    for(int ii=0;ii<16;ii++) W0[ii]=0u;
                    int wpos = 0;
                    // base remainder
                    #pragma unroll
                    for(int j=0;j<rem;j++){
                        int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                        W0[wi] |= ((uint32_t)s_base_rem[j]) << sh; wpos++;
                    }
                    if(binary_nonce){
                        #pragma unroll
                        for(int j=0;j<8;j++){
                            uint8_t by = (uint8_t)((nonce >> (8*j)) & 0xff);
                            int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                            W0[wi] |= ((uint32_t)by) << sh; wpos++;
                        }
                    } else {
                        for(int j=0;j<nonce_len;j++){
                            int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8);
                            W0[wi] |= ((uint32_t)usep[j]) << sh; wpos++;
                        }
                    }
                    // 0x80 and bit length
                    { int wi = wpos >> 2; int sh = 24 - ((wpos & 3) * 8); W0[wi] |= (0x80u << sh); wpos++; }
                    W0[14] = (uint32_t)(bit_len >> 32);
                    W0[15] = (uint32_t)(bit_len & 0xffffffffu);
                    sha256_compress_words(W0, H1);
                } else {
                    // Two-block path: assemble both blocks in words
                    uint32_t Wa[16];
                    #pragma unroll
                    for(int ii=0; ii<16; ++ii) Wa[ii]=0u;
                    int wposA = 0;
                    // base remainder
                    #pragma unroll
                    for(int j=0;j<rem;j++){
                        int wi = wposA >> 2; int sh = 24 - ((wposA & 3) * 8);
                        Wa[wi] |= ((uint32_t)s_base_rem[j]) << sh; wposA++;
                    }
                    int cap = 64 - rem;
                    int n1 = nonce_len < cap ? nonce_len : cap;
                    if(binary_nonce){
                        #pragma unroll
                        for(int j=0;j<n1;j++){
                            uint8_t by = (uint8_t)((nonce >> (8*j)) & 0xff);
                            int wi = wposA >> 2; int sh = 24 - ((wposA & 3) * 8);
                            Wa[wi] |= ((uint32_t)by) << sh; wposA++;
                        }
                    } else {
                        for(int j=0;j<n1;j++){
                            int wi = wposA >> 2; int sh = 24 - ((wposA & 3) * 8);
                            Wa[wi] |= ((uint32_t)usep[j]) << sh; wposA++;
                        }
                    }
                    int wrote80A = 0;
                    if(wposA < 64){ int wi = wposA >> 2; int sh = 24 - ((wposA & 3) * 8); Wa[wi] |= (0x80u << sh); wrote80A = 1; }
                    sha256_compress_words(Wa, H1);

                    uint32_t Wb[16];
                    #pragma unroll
                    for(int ii=0; ii<16; ++ii) Wb[ii]=0u;
                    int wposB = 0;
                    int n2 = nonce_len - n1;
                    if(binary_nonce){
                        #pragma unroll
                        for(int j=0;j<n2;j++){
                            uint8_t by = (uint8_t)((nonce >> (8*(j+n1))) & 0xff);
                            int wi = wposB >> 2; int sh = 24 - ((wposB & 3) * 8);
                            Wb[wi] |= ((uint32_t)by) << sh; wposB++;
                        }
                    } else {
                        for(int j=0;j<n2;j++){
                            int wi = wposB >> 2; int sh = 24 - ((wposB & 3) * 8);
                            Wb[wi] |= ((uint32_t)usep[n1+j]) << sh; wposB++;
                        }
                    }
                    if(!wrote80A){ int wi = wposB >> 2; int sh = 24 - ((wposB & 3) * 8); Wb[wi] |= (0x80u << sh); wposB++; }
                    Wb[14] = (uint32_t)(bit_len >> 32);
                    Wb[15] = (uint32_t)(bit_len & 0xffffffffu);
                    sha256_compress_words(Wb, H1);
                }

                uint32_t H2[8];
                sha256_32B_from_words(H1, H2);
                uint32_t lz = count_leading_zeros_words(H2);

                if(lz > local_best_lz){ local_best_lz = lz; local_best_nonce = nonce; }
                if(enable_live){
                    unsigned int snapshot = s_g_best;
                    unsigned int lz_u = (unsigned int)lz;
                    if(lz_u > snapshot){
                        unsigned int old = atomicMax((unsigned int*)g_best_lz_live, lz_u);
                        if(lz_u > old){ *g_best_nonce_live = nonce; s_g_best = lz_u; }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write per-block best via warp/block reduction
    warp_reduce_max(local_best_lz, local_best_nonce);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp_id = threadIdx.x >> 5;
    if(lane == 0){ s_lz[warp_id] = local_best_lz; s_nonce[warp_id] = local_best_nonce; }
    __syncthreads();
    if(warp_id == 0){
        uint32_t v_lz = (threadIdx.x < (blockDim.x + 31)/32) ? s_lz[lane] : 0u;
        uint64_t v_no = (threadIdx.x < (blockDim.x + 31)/32) ? s_nonce[lane] : 0ull;
        warp_reduce_max(v_lz, v_no);
        if(lane == 0){ block_best_lz[blockIdx.x] = v_lz; block_best_nonce[blockIdx.x] = v_no; }
    }
}

// ASCII-only persistent kernel (no binary branches), using odometer and fast itoa
extern "C" __global__
void double_sha256_persistent_kernel_ascii(const uint8_t* __restrict__ base_message, size_t base_len,
    uint64_t start_nonce, uint64_t total_nonce, uint32_t /*binary_nonce_dummy*/,
    unsigned long long* next_index, uint32_t chunk_size, uint32_t iters_per_thread,
    uint32_t enable_live, uint32_t odometer_ascii,
    uint64_t* block_best_nonce, uint32_t* block_best_lz,
    uint32_t* g_best_lz_live, uint64_t* g_best_nonce_live,
    const volatile uint32_t* stop_flag)
{
    extern __shared__ uint8_t sdata[];
    const int warps = (blockDim.x + 31) >> 5;
    uint32_t* s_lz = (uint32_t*)sdata;
    uint64_t* s_nonce = (uint64_t*)(s_lz + warps);
    uint32_t* s_mid = (uint32_t*)(s_nonce + warps);
    uint8_t* s_base_rem = (uint8_t*)(s_mid + 8);

    // midstate + base remainder
    const size_t full_blocks = base_len / 64;
    const int rem = (int)(base_len % 64);
    if(threadIdx.x == 0){
        uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        for(size_t b=0; b<full_blocks; ++b){
            const uint8_t* blk = base_message + b*64;
            sha256_compress_block(blk, H);
        }
        #pragma unroll
        for(int i=0;i<8;i++) s_mid[i]=H[i];
        #pragma unroll
        for(int i=0;i<rem;i++) s_base_rem[i] = base_message[full_blocks*64 + i];
    }
    __syncthreads();

    uint32_t local_best_lz = 0;
    uint64_t local_best_nonce = 0;

    __shared__ unsigned long long block_start;
    __shared__ unsigned int s_g_best;
    for(;;){
        if(*stop_flag){ break; }
        if(threadIdx.x == 0){
            block_start = atomicAdd(next_index, (unsigned long long)chunk_size);
            if(enable_live){ s_g_best = *g_best_lz_live; }
        }
        __syncthreads();
        if(block_start >= total_nonce) break;
        unsigned long long block_end = block_start + (unsigned long long)chunk_size;
        if(block_end > total_nonce) block_end = total_nonce;

        for(unsigned long long seg = block_start + (unsigned long long)threadIdx.x * (unsigned long long)iters_per_thread;
            seg < block_end; seg += (unsigned long long)blockDim.x * (unsigned long long)iters_per_thread){

            unsigned long long idx0 = seg; if(idx0 >= block_end) break;
            uint64_t nonce0 = start_nonce + (uint64_t)idx0;
            char dec0[21]; int len0 = itoa_u64_fast4(nonce0, dec0);
            char dec1[21]; int len1 = 0;
            if(iters_per_thread > 1){ unsigned long long idx1 = seg + 1ull; if(idx1 < block_end){ uint64_t nonce1 = start_nonce + (uint64_t)idx1; len1 = itoa_u64_fast4(nonce1, dec1); } }

            #pragma unroll 4
            for(uint32_t iter=0; iter<4; ++iter){
                if(iter >= iters_per_thread) break;
                unsigned long long idx = seg + (unsigned long long)iter;
                if(idx >= block_end) break;
                uint64_t nonce = start_nonce + (uint64_t)idx;

                // choose stream 0/1
                const char* decp; int declen;
                if(odometer_ascii && iters_per_thread > 1){
                    if((iter & 1u)==0u){
                        // stream 0
                        if(iter>0){ int i=len0-1; for(; i>=0; --i){ if(dec0[i] != '9'){ dec0[i] += 1; break; } dec0[i]='0'; } if(i<0){ for(int j=len0;j>0;--j) dec0[j]=dec0[j-1]; dec0[0]='1'; len0++; } }
                        decp = dec0; declen = len0;
                    } else {
                        // stream 1
                        if(iter>1){ int i=len1-1; for(; i>=0; --i){ if(dec1[i] != '9'){ dec1[i] += 1; break; } dec1[i]='0'; } if(i<0){ for(int j=len1;j>0;--j) dec1[j]=dec1[j-1]; dec1[0]='1'; len1++; } }
                        decp = dec1; declen = (len1>0?len1:itoa_u64_fast4(nonce, (char*)dec1)); if(len1==0) len1 = declen;
                    }
                } else if(odometer_ascii){
                    // single stream odometer
                    if(iter>0){ int i=len0-1; for(; i>=0; --i){ if(dec0[i] != '9'){ dec0[i] += 1; break; } dec0[i]='0'; } if(i<0){ for(int j=len0;j>0;--j) dec0[j]=dec0[j-1]; dec0[0]='1'; len0++; } }
                    decp = dec0; declen = len0;
                } else {
                    // fallback fresh itoa
                    char tmpb_local[20];
                    declen = itoa_u64_fast4(nonce, (char*)tmpb_local);
                    decp = tmpb_local;
                }

                // First hash tail build: assemble words
                uint32_t H1[8];
                #pragma unroll
                for(int i2=0;i2<8;i2++) H1[i2] = s_mid[i2];

                int pos = 0; int nonce_len = declen; int cap = 64 - rem;
                if(pos <= 55){ /* just to satisfy compiler */ }
                // One-block or two-block paths
                if(rem + nonce_len <= 55){
                    uint32_t W0[16];
                    #pragma unroll
                    for(int ii=0;ii<16;ii++) W0[ii]=0u;
                    int wpos=0; for(int j=0;j<rem;j++){ int wi=wpos>>2; int sh=24-((wpos&3)*8); W0[wi]|=((uint32_t)s_base_rem[j])<<sh; wpos++; }
                    for(int j=0;j<nonce_len;j++){ int wi=wpos>>2; int sh=24-((wpos&3)*8); W0[wi]|=((uint32_t)decp[j])<<sh; wpos++; }
                    { int wi=wpos>>2; int sh=24-((wpos&3)*8); W0[wi]|=(0x80u<<sh); }
                    uint64_t bit_len = (uint64_t)(base_len + (uint64_t)nonce_len) * 8ull;
                    W0[14]=(uint32_t)(bit_len>>32); W0[15]=(uint32_t)bit_len;
                    sha256_compress_words(W0, H1);
                } else {
                    // Two-block
                    uint32_t Wa[16];
                    #pragma unroll
                    for(int ii=0;ii<16;ii++) Wa[ii]=0u; int wA=0;
                    for(int j=0;j<rem;j++){ int wi=wA>>2; int sh=24-((wA&3)*8); Wa[wi]|=((uint32_t)s_base_rem[j])<<sh; wA++; }
                    int n1 = nonce_len < cap ? nonce_len : cap;
                    for(int j=0;j<n1;j++){ int wi=wA>>2; int sh=24-((wA&3)*8); Wa[wi]|=((uint32_t)decp[j])<<sh; wA++; }
                    int wrote80A=0; if(wA<64){ int wi=wA>>2; int sh=24-((wA&3)*8); Wa[wi]|=(0x80u<<sh); wrote80A=1; }
                    sha256_compress_words(Wa, H1);
                    uint32_t Wb[16];
                    #pragma unroll
                    for(int ii=0;ii<16;ii++) Wb[ii]=0u; int wB=0; int n2=nonce_len-n1;
                    for(int j=0;j<n2;j++){ int wi=wB>>2; int sh=24-((wB&3)*8); Wb[wi]|=((uint32_t)decp[n1+j])<<sh; wB++; }
                    if(!wrote80A){ int wi=wB>>2; int sh=24-((wB&3)*8); Wb[wi]|=(0x80u<<sh); wB++; }
                    uint64_t bit_len = (uint64_t)(base_len + (uint64_t)nonce_len) * 8ull;
                    Wb[14]=(uint32_t)(bit_len>>32); Wb[15]=(uint32_t)bit_len;
                    sha256_compress_words(Wb, H1);
                }

                // Second hash from words
                uint32_t H2[8]; sha256_32B_from_words(H1, H2);
                uint32_t lz = count_leading_zeros_words(H2);
                if(lz > local_best_lz){ local_best_lz = lz; local_best_nonce = nonce; }
                if(enable_live){ unsigned int snap=s_g_best; unsigned int lzu=(unsigned)lz; if(lzu>snap){ unsigned old=atomicMax((unsigned*)g_best_lz_live,lzu); if(lzu>old){ *g_best_nonce_live=nonce; s_g_best=lzu; }}}
            }
        }
        __syncthreads();
    }

    // Write per-block best
    warp_reduce_max(local_best_lz, local_best_nonce);
    unsigned lane = threadIdx.x & 31u; unsigned warp_id = threadIdx.x >> 5;
    if(lane==0){ s_lz[warp_id]=local_best_lz; s_nonce[warp_id]=local_best_nonce; }
    __syncthreads();
    if(warp_id==0){ uint32_t v_lz = (threadIdx.x < (blockDim.x+31)/32) ? s_lz[lane] : 0u; uint64_t v_no = (threadIdx.x < (blockDim.x+31)/32) ? s_nonce[lane] : 0ull; warp_reduce_max(v_lz,v_no); if(lane==0){ block_best_lz[blockIdx.x]=v_lz; block_best_nonce[blockIdx.x]=v_no; } }
}

// Kernel 2: reduce per-block results to a single best (lz, nonce)
extern "C" __global__
void reduce_best_kernel(const uint32_t* __restrict__ block_lz,
                        const uint64_t* __restrict__ block_nonce,
                        uint32_t n,
                        uint32_t* best_lz_out,
                        uint64_t* best_nonce_out)
{
    extern __shared__ uint8_t sdata2[];
    // warp 数量
    const int warps = (blockDim.x + 31) >> 5;
    uint32_t* s_lz = (uint32_t*)sdata2;
    uint64_t* s_nonce = (uint64_t*)(s_lz + warps);

    unsigned tid = threadIdx.x;
    // 每线程以 stride 步长扫描输入，找本线程局部最优
    uint32_t my_lz = 0;
    uint64_t my_nonce = 0;
    for(uint32_t i = tid; i < n; i += blockDim.x){
        uint32_t lz = block_lz[i];
        uint64_t no = block_nonce[i];
        if(lz > my_lz){ my_lz = lz; my_nonce = no; }
    }

    // warp 内归约
    warp_reduce_max(my_lz, my_nonce);
    unsigned lane = tid & 31u;
    unsigned warp_id = tid >> 5;
    if(lane == 0){
        s_lz[warp_id] = my_lz;
        s_nonce[warp_id] = my_nonce;
    }
    __syncthreads();

    if(warp_id == 0){
        uint32_t v_lz = (tid < (blockDim.x + 31)/32) ? s_lz[lane] : 0u;
        uint64_t v_no = (tid < (blockDim.x + 31)/32) ? s_nonce[lane] : 0ull;
        warp_reduce_max(v_lz, v_no);
        if(lane == 0){
            *best_lz_out = v_lz;
            *best_nonce_out = v_no;
        }
    }
}
