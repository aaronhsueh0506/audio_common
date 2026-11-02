> [索引](c_user_manual_zh_TW.md) | [上一部分](c_user_manual_zh_TW_01_modules_core.md) | [下一部分](c_user_manual_zh_TW_03_checklist.md) <!-- split-nav -->
### 5.5 `simd_kernels.h` — per-bin 微核心（純標頭）

#### 何時使用

當你自己的 per-bin／per-sample 迴圈是熱點、而下表剛好有形狀一致的核心時。
每個核心都有兩個進入點：`sk_<name>()`（在支援的平台上走 NEON）與 `sk_<name>_scalar()`（永遠是純量）。**兩者對任何有限輸入產生逐位元相同的結果**，所以可以在效能與可攜性之間切換而不改變數值輸出。

如果你的迴圈形狀跟下表都不一樣，就不要硬套——直接寫你自己的純量迴圈。神經網路前後處理另有 `simd_kernel_nn.h` 的 `skn_*` 核心（同樣的 scalar 雙胞胎與 `SIMD=0` 規則），不在本手冊逐一列出。

#### 生命週期

純標頭，`static inline`，沒有 handle、沒有初始化、沒有釋放，不需要連結封存檔。
需要 `-I<common>/include`（`simd_kernels.h` 會 include `ac_complex.h`）與 `-I<common>`（`utility/` 標頭以 `utility/xxx.h` 引用），並帶 `-ffp-contract=off -fno-math-errno`（見第 3.5 節）。

#### 逐函式簽章與呼叫時機

```c
/* 建置狀態 */
#define SK_HAVE_NEON   /* 1 = 編進 NEON 實作，0 = 全部走純量 */

void sk_ema_f32(float *state, const float *x, float alpha, float beta, int n);
void sk_capply_gain_f32(Complex *out, const Complex *z, const float *g, int n);
void sk_cadd_f32(Complex *out, const Complex *a, const Complex *b, int n);
void sk_sq_scale_f32(const float *x, float scale, float *out, int n);
void sk_min_f32(float *out, const float *a, const float *b, int n);
void sk_clip_f32(float *x, float lo, float hi, int n);
void sk_fast_sqrt_f32(const float *x, float *out, int n);
void sk_fast_exp_f32(const float *x, float *out, int n);
void sk_fast_exp_neg_f32(const float *x, float *out, int n);
void sk_fast_log_f32(const float *x, float *out, int n);
void sk_exp1_approx_f32(const float *x, float *out, int n);
void sk_noise_update_f32(float *noise_psd, const float *spp,
                         const float *power,
                         float alpha_d, float bb_scale, int n);
void sk_csub_f32(Complex *out, const Complex *a, const Complex *b, int n);
void sk_wola_accumulate_f32(float *acc, const float *x, const float *w, int n);
void sk_cmag_f32(float *out, const Complex *x, int n, float eps);
void sk_asym_ema_f32(float *s, const float *x, int n, float a_up, float a_down);
void sk_scale_f32(float *out, const float *x, float g, int n);
void sk_linear_to_db_f32(float *out, const float *x, int n);          /* 只有純量實作 */
void sk_clip_scale_to_s16_arr(int16_t *out, const float *in, int n);
int  sk_argmax_f32(const float *x, int n);
int  sk_argmax_sq_f32(const float *h, int n);
int  sk_argmax_i32(const int *a, int n);                                /* 只有純量實作，n >= 1 */
void sk_cmac_f32(Complex *acc, const Complex *w, const Complex *x, int n);
void sk_cmac_fma_ring_f32(Complex *acc, const Complex *W, const Complex *X_buf,
                          int curr_p, int n_part, int K);
/* 除了上面標註「只有純量」的兩個，其餘都有對應的 sk_<name>_scalar(...)，參數完全相同 */
```

| 核心 | 做什麼 | 什麼時候用 |
|---|---|---|
| `sk_ema_f32` | `state[i] = alpha*state[i] + beta*x[i]` | 單一標量 alpha 的指數平滑 |
| `sk_capply_gain_f32` | `out[i] = z[i] * g[i]`（實數增益乘上複數的實部與虛部） | 對頻譜逐 bin 套實數增益 |
| `sk_cadd_f32` | `out[i] = a[i] + b[i]`（複數逐項相加） | 頻譜相加 |
| `sk_sq_scale_f32` | `out[i] = (x[i]*x[i]) * scale` | 平方後縮放（例如均方能量） |
| `sk_min_f32` | `out[i] = (a[i] < b[i]) ? a[i] : b[i]` | 逐項取小 |
| `sk_clip_f32` | 就地夾在 `[lo, hi]`（先比下界再比上界） | 逐項夾限 |
| `sk_fast_sqrt_f32` | `fast_math.h` 的 `fast_sqrt` 陣列版 | 大量開根號 |
| `sk_fast_exp_f32` / `sk_fast_exp_neg_f32` | `fast_exp` / `fast_exp_neg` 陣列版 | 大量指數 |
| `sk_fast_log_f32` | `fast_log` 陣列版 | 大量對數 |
| `sk_exp1_approx_f32` | `exp1_approx` 陣列版 | 大量 E1 近似 |
| `sk_noise_update_f32` | `noise_psd[i] = a*noise_psd[i] + (1-a)*power[i]`，其中 `a = alpha_d + (1-alpha_d)*spp[i]*bb_scale`（語音存在機率越高 `a` 越接近 1，估計值越不動） | 追蹤噪聲功率譜（噪聲 PSD），平滑係數隨每個 bin 的語音存在機率變動 |
| `sk_csub_f32` | `out[i] = a[i] - b[i]`（複數逐項相減） | 頻譜相減 |
| `sk_wola_accumulate_f32` | `acc[i] += x[i]*w[i]` | overlap-add 累加（三個陣列不可別名） |
| `sk_cmag_f32` | `out[i] = x[i].r² + x[i].i² + eps`（**平方幅度加下限，沒有開根號**，名字容易誤導） | 功率譜加 floor |
| `sk_asym_ema_f32` | `s[i]` 就地更新：`x[i] < s[i]` 用 `a_down`，否則用 `a_up`（嚴格小於） | 上升與下降係數不同的 EMA |
| `sk_scale_f32` | `out[i] = x[i]*g`（單一 broadcast 增益） | 整段乘常數 |
| `sk_linear_to_db_f32` | `out[i] = 10*log10f(x[i])`，**只有純量**（libm `log10f`） | 功率轉 dB |
| `sk_clip_scale_to_s16_arr` | float 夾到 ±1、乘 32767、向零截斷成 int16；**不處理 NaN**（呼叫端須先排除）；單 quad 版 `sk_clip_scale_to_s16` 只在 NEON、回傳 `int16x4_t` | 輸出到 PCM16 |
| `sk_argmax_f32`／`sk_argmax_sq_f32`／`sk_argmax_i32` | 回傳最大值（平方值／int32）的**第一個**索引 | 找峰值 |
| `sk_cmac_f32`／`sk_cmac_fma_ring_f32` | 複數乘加 `acc[i] += w[i]*x[i]`；後者用 FMA 形式沿 `n_part` 個分割累加，`x` 取環形緩衝；**兩者不可互換**（捨入不同） | 濾波器乘加 |

**選 `sk_*` 還是 `sk_*_scalar`？** 平常一律呼叫不帶後綴的版本，它會自動依建置設定選路。只有在要做「NEON vs 純量」對照驗證時才直接呼叫 `_scalar` 版本。

**數值定義域**：`sk_fast_*` 與 `sk_exp1_approx_f32` 的每個元素行為與 `fast_math.h` 的同名函式完全一致（含定義域邊界值），見[第 5.6 節](c_user_manual_zh_TW_02_modules_util_io.md#56-fast_mathh--快速數學近似純標頭)。

#### 緩衝區大小規則

- 所有陣列參數的長度都是 `n`，單位是元素數（`float` 或 `Complex`），不是位元組。
- `Complex` 陣列 `n` 個元素佔 `n * 2 * sizeof(float)` 位元組。
- **`n` 不需要是 4 的倍數**：每個核心的 NEON 路徑處理完 4 的倍數後，尾端剩餘元素會走與純量版本完全相同的逐元素程式碼。
- 沒有對齊要求：核心使用不要求對齊的載入／儲存。

#### 就地／別名限制

**這是本模組最容易出錯的地方。** 除了下列明確支援的情況，所有指標都假設**互不重疊**。

| 核心 | 支援的就地形式 |
|---|---|
| `sk_clip_f32` | 本來就是就地（只有一個 `x` 參數） |
| `sk_ema_f32` | `state` 是就地讀寫；`x` 必須與 `state` 不重疊 |
| `sk_capply_gain_f32`、`sk_scale_f32`、`sk_csub_f32` | `sk_capply_gain_f32` 支援 `out == z`、`sk_scale_f32` 支援 `out == x`、`sk_csub_f32` 支援 `out == a`（完全相同指標） |
| `sk_fast_exp_f32`、`sk_fast_exp_neg_f32`、`sk_fast_log_f32`、`sk_exp1_approx_f32` | 支援 `out == x`（完全相同指標） |
| `sk_noise_update_f32`、`sk_asym_ema_f32`、`sk_cmac_f32`、`sk_cmac_fma_ring_f32` | `noise_psd`／`s`／`acc` 是就地讀寫；其餘輸入必須與它不重疊 |
| 其他所有核心（含 `sk_fast_sqrt_f32`、`sk_min_f32`、`sk_cadd_f32`、`sk_sq_scale_f32`、`sk_wola_accumulate_f32`、`sk_cmag_f32`） | **不支援任何別名** |

**部分重疊（`out == x + k`，`k != 0`）在所有核心一律不支援**，即使某次執行看起來正常。

#### 錯誤回傳

**沒有錯誤通道。** 所有核心都是 `void`，不檢查 NULL 指標、不檢查 `n` 的範圍。傳入 NULL 會直接當機；傳入負的 `n` 會讓迴圈不執行。呼叫端必須自己保證參數正確。

#### 常見錯誤

1. include 了這個標頭但 TU 沒加 `-ffp-contract=off`，導致數值與封存檔內的同名邏輯不一致。
2. 封存檔用 `SIMD=0` 建、自己的 TU 沒定義 `SIMD_KERNELS_FORCE_SCALAR`（或反過來），兩邊走不同路徑。
3. 用了未列在支援表中的別名形式（尤其是 `sk_fast_sqrt_f32` 的 `out == x`）。
4. 把 `n` 當成位元組數。
5. 對 `Complex` 陣列傳入 `n * 2`（誤把實部虛部各算一個元素）。

---

### 5.6 `fast_math.h` — 快速數學近似（純標頭）

#### 何時使用

在逐樣本／逐 bin 的熱迴圈裡需要 `exp`、`log`、`log10`、`sqrt` 或指數積分 `E1` 時。它們是**有限定義域的近似**，不是 libm 的替代品：精度換速度。
若你需要標準 libm 的精度與 IEEE 特殊值語意（例如 `sqrtf(-1)` 要得到 NaN），就直接用 libm。

#### 生命週期

純標頭，`static inline`，沒有 handle、沒有狀態，不需要連結封存檔。

#### 逐函式簽章與呼叫時機

```c
/* 常數 */
#define FM_LN2      0.693147180559945f
#define FM_LOG10E   0.4342944819032518f
#define FM_LN10     2.302585092994046f
#define FM_EPSILON  1e-10f

static inline float fast_exp(float x);
static inline float fast_exp_neg(float x);   /* 等同 exp(-x)，針對 x >= 0 最佳化 */
static inline float fast_log(float x);
static inline float fast_log10(float x);
static inline float fast_sqrt(float v);
static inline float exp1_approx(float v);    /* Martin et al. 2004 式 (17) 的 E1 三段近似 */

static inline float clip_f(float x, float min_val, float max_val);
static inline float max_f(float a, float b);
```

| 函式 | 有效輸入範圍 | 超出範圍時的行為 |
|---|---|---|
| `fast_exp(x)` | `x ∈ [-16, 16]` | `x < -16` 或 `x` 為 NaN → `0.0f`；`x > 16`（含 `+Inf`）→ `8.8861105e+06f` |
| `fast_exp_neg(x)` | `x ∈ [0, 16]` | `x <= 0` → `1.0f`；`x >= 16` → `0.0f` |
| `fast_log(x)` | 正規化（normal）的有限正數；production 呼叫端會 clamp 到 `FLT_MIN` 以上 | `x <= 0`（含 `±0`）或 NaN → `-1e10f`（代表 −∞）；subnormal 不在精度保證範圍 |
| `fast_log10(x)` | 同 `fast_log` | 同 `fast_log`，再乘上 `FM_LOG10E` |
| `fast_sqrt(v)` | `v > 0` 的有限值 | `v <= 0`（含 `-0.0f`）或 NaN → `0.0f`（正零）；`+Inf` 在 ARM32／ARM64 的 sqrtf 路徑回 `+Inf` |
| `exp1_approx(v)` | `v > 0` | `v <= FM_EPSILON` 會先被夾成 `FM_EPSILON` 再計算 |
| `clip_f` / `max_f` | 任意有限值 | 無特殊處理，就是單純的比較選擇 |

**重點：這些函式對非有限輸入回傳的是「定義域邊界常數」，不是 IEEE 語意的結果。**
例如 `fast_sqrt(NaN)` 回 `0.0f`（不是 NaN）、`fast_log(-1.0f)` 回 `-1e10f`（不是 NaN）、`fast_exp(+Inf)` 回一個有限的飽和值（不是 `+Inf`）。這些值是確定且穩定的，但**不具數學意義**。正確做法是在資料進入 DSP 之前就把非有限樣本清掉（`wav_read_float()` 在讀檔端已經做了這件事），而不是依賴這些邊界值。

production `fast_log` 以 IEEE-754 range reduction 加 degree-4 minimax 多項式實作（Horner，各目標同樣的分離乘加順序，不用 FMA）；selftest 對全部 `2^23` 個 normalized mantissa 逐一檢查，並以固定步距掃過所有正規化的正數 float，最大絕對誤差契約為 `< 1e-4`。`fast_sqrt` 在 AArch64 使用 `FSQRT`，ARM32 VFP 使用 `VSQRT`（正確捨入，與 `sqrtf` 逐位元相同）；ARM32 soft-float 則呼叫 libm。A53/A73 硬體 sqrt codegen audit 要求沒有 `sqrtf` errno fallback；非 ARM 目標保留兩次 Newton iteration 的 portable fallback。

編譯期巨集 `USE_STANDARD_MATH` 會把 `fast_*` 全部換成 libm（非有限輸入的結果隨之改為 IEEE 語意，只用於對照）；`USE_OPTIMIZED_E1` 只改 `exp1_approx` 的分支順序，結果必須與預設順序逐位元相同（`make selftest` 兩種都跑）。

`exp1_approx` 的數學目標是 Ephraim–Malah MMSE-LSA 增益裡的精確指數積分；production 三段係數則固定採用 Martin、Malah、Cox、Accardi 2004 論文式 (17) 的低複雜度近似，不把它當成可自由 retune 的 NR 參數。

#### 使用方式

```c
#include "utility/fast_math.h"

for (int k = 0; k < n_freqs; ++k) {
    float mag = fast_sqrt(power[k]);
    float db  = 20.0f * fast_log10(mag + FM_EPSILON);   /* 先加 epsilon 避開 0 */
    gain[k]   = clip_f(gain[k], 0.05f, 1.0f);
}
```

#### 緩衝區大小規則

不適用——全部都是純量對純量的函式。要對整個陣列做同樣運算時，請改用 `simd_kernels.h` 的 `sk_fast_*` 陣列版本（[第 5.5 節](c_user_manual_zh_TW_02_modules_util_io.md#55-simd_kernelsh--per-bin-微核心純標頭)），它們的每元素行為與這裡完全一致。

#### 就地／別名限制

不適用（傳值、傳回值，沒有指標）。

#### 錯誤回傳

**沒有錯誤通道。** 所有函式都回傳 `float`，永遠不會失敗、不會 trap、不會產生未定義行為；超出定義域時回傳上表列出的固定邊界值。呼叫端無法從回傳值區分「真的算出這個值」與「輸入超出定義域」，所以請自己保證輸入有效。

#### 常見錯誤

1. 期待 `fast_sqrt(negative)` 回 NaN（實際回 `0.0f`），因此漏掉本來能靠 NaN 察覺的上游發散問題。
2. 對 `fast_log(0.0f)` 的結果做算術，得到 `-1e10f` 這個巨大有限值污染後續計算。忘了加 epsilon。
3. 輸入超過 `[-16, 16]` 還期待 `fast_exp` 給出正確量級。
4. 在需要位元級比對的路徑上，一邊用 `fast_exp()`、另一邊用 `expf()`，兩者結果不同。
5. 逐元素呼叫 `fast_exp()` 跑滿一整個陣列，而不是用 `sk_fast_exp_f32()`。

---

### 5.7 `wav_io.h` — WAV 讀寫（純標頭）

#### 何時使用

離線工具、測試程式、批次轉檔——需要從磁碟讀入音訊或把處理結果寫成檔案時。
**不適合即時／嵌入式路徑**：它使用 `stdio`、`malloc`，而且每個樣本一次 `fread`／`fwrite`。

#### 生命週期

純標頭（`static inline`），不需要連結封存檔，但物件本身是 `malloc` 出來的，有明確的擁有權規則。

**讀取端**

```c
WavReader* r = wav_open_read("input.wav");
if (!r) { /* 開檔失敗或格式不被接受 */ }
/* ... 反覆呼叫 wav_read_float ... */
wav_close_read(r);          /* 一定要呼叫；NULL 安全 */
```

**寫入端**

```c
WavWriter* w = wav_open_write("output.wav", 16000, 1);
if (!w) { /* 開檔失敗或參數不合法 */ }
/* ... 反覆呼叫 wav_write_float ... */
if (wav_finalize_write(w) != 0) { /* 檔案內容不可信 */ }
/* w 已被消耗，不可再使用 */
```

**擁有權**：`wav_finalize_write()` **無論成功或失敗都會消耗 `w`**（關檔並釋放結構）。呼叫之後不可以再解參考 `w`，也不可以再對同一個指標呼叫 `wav_finalize_write()` 或 `wav_close_write()`。

#### 逐函式簽章與呼叫時機

```c
typedef struct { int sample_rate; int channels; int bits_per_sample;
                 int num_samples; int is_float; } WavInfo;

typedef struct { FILE* fp; WavInfo info; long data_start; int samples_read;
                 uint64_t nonfinite_sanitized; } WavReader;

typedef struct { FILE* fp; WavInfo info; long data_start;
                 uint64_t samples_written; uint64_t nonfinite_sanitized;
                 int write_error; } WavWriter;

WavReader* wav_open_read(const char* path);
int        wav_read_float(WavReader* r, float* buf, int n);
void       wav_close_read(WavReader* r);

WavWriter* wav_open_write(const char* path, int sample_rate, int channels);
void       wav_write_float(WavWriter* w, const float* buf, int n);
int        wav_finalize_write(WavWriter* w);   /* 建議在新程式碼使用 */
void       wav_close_write(WavWriter* w);      /* 相容包裝，吞掉結果 */
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `wav_open_read` | 開始讀檔時 | 成功後由 `r->info` 取得取樣率／通道數／總樣本數 |
| `wav_read_float` | 反覆呼叫直到回傳值 `< n` | `buf` 至少 `n` 個 `float`；回傳實際讀到的樣本數 |
| `wav_close_read` | 讀完 | NULL 安全 |
| `wav_open_write` | 開始寫檔時 | `sample_rate` 與 `channels` 都必須 `> 0` |
| `wav_write_float` | 每個區塊 | `buf` 為 `n` 個 `float`；`n <= 0` 直接 return |
| `wav_finalize_write` | 寫完，且**只呼叫一次** | 回 `0` 成功、`-1` 失敗；一定會消耗 `w` |
| `wav_close_write` | 相容包裝 | 等同 `wav_finalize_write()` 但丟棄回傳值 |

**接受的檔案格式（`wav_open_read`）**

只接受下列兩種組合，其他一律在開檔時回 `NULL`：

| `audio_format` | `bits_per_sample` | 意義 |
|---|---|---|
| `1` | `16` | PCM 16-bit 整數 |
| `3` | `32` | IEEE float32 |

其他限制（任一不符即拒絕）：

- `channels` 必須在 `[1, 8]`。
- `sample_rate` 必須在 `[1, 384000]`。
- `fmt` chunk 宣告的大小必須 `>= 16`。
- 檔案內宣告的 `block_align`（非零時）必須等於 `channels * bytes_per_sample`；`byte_rate`（非零時）必須等於 `sample_rate * block_align`。
- 任何 chunk 宣告的大小若超出檔案實際剩餘位元組數，整個檔案被拒絕。
- 樣本總數必須放得下 `int`。

24-bit PCM、32-bit 整數 PCM、A-law／μ-law、`WAVE_FORMAT_EXTENSIBLE` 等格式**不被支援**，會在開檔時就被拒絕。

**Mono 讀取行為（重要）**

`wav_read_float()` **永遠只回傳第一個通道**。對多通道檔案，它讀第 0 通道的樣本，然後跳過該 frame 其餘通道。
- `n` 的單位是**輸出樣本數**，也就是 frame 數。
- 回傳值是實際寫進 `buf` 的樣本數。
- **這個 API 沒有辦法取得第 2 個以上的通道。** 需要多通道請自己開檔解析，或在上游先分軌。

PCM16 來源會被轉成 float（除以 `32768.0f`）。float32 來源若含 NaN 或 ±Inf，會被替換成 `0.0f`，同時 `r->nonfinite_sanitized` 累加。

**兩種寫入風格**

由編譯期巨集 `WAV_IO_WRITER_STYLE` 選擇，必須在 include 這個標頭**之前**定義：

| 值 | 巨集名 | 行為 |
|---|---|---|
| `1`（未設定時的預設） | `WAV_IO_WRITER_AEC` | 預設輸出 PCM16；量化為「先乘 `32768.0f`，再 ±0.5 後截斷」（四捨五入、遠離零），並飽和到 `[-32768, 32767]`。若環境變數 `AEC_OUT_FLOAT` 的值為 `1`，改輸出未量化的 IEEE float32 |
| `2` | `WAV_IO_WRITER_NR` | 永遠輸出 PCM16；量化為「乘 `32767.0f` 後直接截斷」（不做四捨五入），同樣飽和到 `[-32768, 32767]`。沒有 float32 路徑 |

```c
#define WAV_IO_WRITER_STYLE WAV_IO_WRITER_NR
#include "wav_io.h"
```

兩種風格輸出的位元組**不相同**。做位元級回歸比對時，兩邊必須用同一種風格。

PCM16 路徑會先把非有限樣本換成 `0.0f`（並累加 `w->nonfinite_sanitized`），再夾到 `[-1, 1]`，再量化。
float32 輸出路徑（`WAV_IO_WRITER_AEC` + `AEC_OUT_FLOAT=1`）是**原始位元直通，不做任何清理**。

**可觀測欄位**

這三個欄位可以直接讀，用來判斷資料品質與 I/O 狀況：

| 欄位 | 意義 |
|---|---|
| `WavReader.nonfinite_sanitized` | 讀入時被換成 `0.0f` 的非有限 float32 樣本數。PCM16 來源永遠是 `0` |
| `WavWriter.nonfinite_sanitized` | 寫出前在 PCM16 路徑被換成 `0.0f` 的樣本數。float32 輸出模式永遠是 `0` |
| `WavWriter.write_error` | `wav_write_float()` 內部發生短寫（例如磁碟滿）時被設為 `1`。該次呼叫剩下的樣本不再嘗試寫入 |

`write_error` 也會被折進 `wav_finalize_write()` 的回傳值，所以只檢查回傳值就足夠；直接讀欄位可以在串流中途就發現問題。

#### 緩衝區大小規則

- `wav_read_float(r, buf, n)`：`buf` 至少 `n` 個 `float`。因為只回第一個通道，`buf` 的大小與檔案通道數無關。
- `wav_write_float(w, buf, n)`：`buf` 至少 `n` 個 `float`。多通道時請自己把資料交錯好後整段送，並確保 `n` 是 `frames * channels`。
- 檔案總樣本數（每通道）由 `r->info.num_samples` 取得。

#### 就地／別名限制

不適用——沒有任何函式會就地改寫呼叫端的緩衝區。`wav_read_float()` 只寫 `buf`，`wav_write_float()` 只讀 `buf`。

#### 錯誤回傳

- `wav_open_read`：失敗回 `NULL`（開檔失敗、非 RIFF/WAVE、格式不被接受、標頭與檔案大小不一致等）。失敗時不會留下任何已配置資源。
- `wav_read_float`：回傳實際讀到的樣本數。`r` 或 `buf` 為 NULL 時回 `0`。回傳值 `< n` 代表已到檔尾或讀取失敗。
- `wav_close_read`：`void`，NULL 安全。
- `wav_open_write`：失敗回 `NULL`（`path` 為 NULL、`sample_rate <= 0`、`channels <= 0`、標頭欄位範圍不合法、`fopen` 失敗、佔位標頭寫入失敗）。
- `wav_write_float`：`void`。`w` 或 `buf` 為 NULL、或 `n <= 0` 時直接 return。寫入失敗只反映在 `w->write_error`。
- `wav_finalize_write`：回 `0` 代表標頭寫入與關檔全部成功、且先前沒有任何短寫；回 `-1` 代表任一環節失敗、尺寸病態、或 `w` 為 NULL。**無論回傳什麼，`w` 都已被釋放。**
- `wav_close_write`：`void`，把上面的結果丟掉。新程式碼請改用 `wav_finalize_write()`。

#### 常見錯誤

1. 用 `wav_read_float()` 讀立體聲檔案，卻以為會拿到兩個通道的資料（實際只有第 0 通道）。
2. 呼叫 `wav_finalize_write()` 之後又呼叫 `wav_close_write()`，造成 double free。
3. 用 `wav_close_write()` 而不檢查任何錯誤，磁碟滿時得到一個標頭正確但資料被截斷的檔案。
4. 在 include 之後才定義 `WAV_IO_WRITER_STYLE`，結果拿到預設風格。
5. 餵 24-bit 等不被接受的格式，`wav_open_read()` 回 `NULL` 卻沒檢查。
6. 在即時路徑上使用這個模組。

---

### 5.8 `audio_stft.h` — sqrt-Hann STFT／ISTFT

#### 何時使用

需要 50% overlap 的 sqrt-Hann 組幀時：`AcStft` 每次吃 hop = `fft_size/2` 個樣本、吐加窗 rFFT 幀；`AcIstft` 吃一幀頻譜、吐 hop 個完成的樣本。

只需要窗表時可直接 include `audio_window.h`，呼叫 `ac_window_build(w, n)`；`w` 由呼叫端配置 n 個 float，函式不配置記憶體，也不依賴 FFT。`n <= 0` 時不寫入。窗以 float32 的 periodic 公式計算，與 NumPy／PyTorch 的建窗捨入可能不同。`audio_stft.h` 也 include 這個 API，既有呼叫端仍可直接使用。

#### 生命週期

只有 static 路徑，沒有 heap、沒有 destroy。

```c
AcStftConfig cfg;
ac_stft_config_default(&cfg, 512);                 /* hop = 256 */
size_t sa = ac_stft_get_mem_size(&cfg);            /* 0 = 設定不合法 */
size_t ss = ac_istft_get_mem_size(&cfg);
AcStft  *an  = ac_stft_init(pool_a, sa, &cfg);     /* pool 16-byte 對齊；NULL = 失敗 */
AcIstft *syn = ac_istft_init(pool_s, ss, &cfg);
```

`cfg.fft`／`cfg.window` 為 NULL 時由 pool 自帶；非 NULL 時借用（handle 的 `n_freqs` 必須等於 `fft_size/2+1`，窗表須維持不變且活得比物件久，一支 handle 可被多個物件循序共用）。

#### 逐函式簽章與呼叫時機

```c
void ac_stft_config_default(AcStftConfig* cfg, int fft_size);
size_t   ac_stft_get_mem_size(const AcStftConfig* cfg);                     /* 0 = 設定不合法 */
AcStft*  ac_stft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
size_t   ac_istft_get_mem_size(const AcStftConfig* cfg);
AcIstft* ac_istft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
void ac_stft_reset(AcStft*);   void ac_istft_reset(AcIstft*);               /* 清歷史與幀計數，設定與窗不變 */
const float* ac_stft_window(const AcStft*);                                 /* 使用中的窗表，可借給 AcIstft */
int  ac_stft_get_config(const AcStft*, AcStftConfig*);                      /* 0；NULL 引數回 -1 */
int  ac_istft_get_config(const AcIstft*, AcStftConfig*);
int  ac_stft_push(AcStft*, const float* hop_in, Complex* out);              /* 回傳幀數 0..2 */
int  ac_stft_push_planar(AcStft*, const float* hop_in, float* re, float* im);
void ac_stft_frame_planar(AcStft*, const float* frame, float* re, float* im); /* 呼叫端自備 N 樣本，不碰歷史 */
int  ac_stft_flush_planar(AcStft*, float* re, float* im);                   /* 僅 center=1：尾端 reflect 幀 */
int  ac_istft_push(AcIstft*, const Complex* spec, float* out);              /* 回傳樣本數：hop 或 0 */
int  ac_istft_push_planar(AcIstft*, const float* re, const float* im, float* out);
int  ac_istft_flush(AcIstft*, float* out);                                  /* 僅 ENV_NORM：hop 個尚在 overlap 內的樣本 */
void ac_window_build(float* w, int n);
```

#### 緩衝區大小規則

`center = 1` 時 `ac_stft_push` 的輸出要容納 2 幀（第 f 幀在 `out + f * n_freqs`）；`center = 0` 恆 1 幀。`ac_istft_push` 的輸出要容納 hop 個 float。

#### 就地／別名限制

輸出緩衝區不得與輸入重疊；`ac_istft_push` 不修改輸入頻譜。

#### 錯誤回傳

`get_mem_size` 回 0、`init` 回 NULL 表示設定不合法（`fft_size` 非 2 的冪或不在 [16, 8192]、`center` 非 0／1、`ola` 不是兩個列舉值之一、借用 handle 大小不符）、pool 太小或未對齊。push／flush 遇 NULL 引數回 0。

#### 常見錯誤

- `spec_scale`／`ispec_scale` 恰為 `1.0f` 時略過乘法；要縮放就給 `1.0f` 以外的值。
- 同一個 `AcStft` 不可混用兩種 `center`；`center` 在建立時固定。
- `AC_OLA_ENV_NORM` 的第 1 幀不吐樣本，`center = 1` 的第 1 次 push 不產出幀；呼叫端自行補第 1 個 hop 的零。

---

> [索引](c_user_manual_zh_TW.md) | [上一部分](c_user_manual_zh_TW_01_modules_core.md) | [下一部分](c_user_manual_zh_TW_03_checklist.md) <!-- split-nav -->
