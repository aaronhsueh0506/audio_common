# FFT、重取樣、高通與 pool

[C 整合手冊](../c_user_manual_zh_TW.md)

## fft_wrapper.h · FFT 包裝層

### 何時使用

需要把時域實數訊號轉到頻域、或把頻域轉回時域時。典型用法是每個 hop 做一次 `fft_forward`，在頻域做處理，再做一次 `fft_inverse`。
這一組 API 在 heap 路徑與 static pool 路徑上行為一致。

### 生命週期

兩條路徑，選一條，不要混用。

**Heap 路徑**

```c
FftHandle* fft = fft_create(512);
if (!fft) { /* 失敗處理 */ }
/* ... 每個 hop 呼叫 fft_forward / fft_inverse ... */
fft_destroy(fft);            /* 只能呼叫一次 */
```

**Static pool 路徑（無 heap 的目標平台）**

```c
size_t need = fft_get_mem_size(512);
if (need == 0) { /* fft_size 不合法 */ }

/* buf 必須是 16-byte 對齊、且至少 need 個 byte */
FftHandle* fft = fft_init(buf, need, 512);
if (!fft) { /* 未對齊 / 空間不足 / 尺寸不合法 */ }
/* ... 使用 ... */
fft_destroy(fft);            /* pool 路徑下是真正的 no-op，重複呼叫也安全 */
```

`fft_get_mem_size()` 回報的位元組數**已包含** NE10 內部的 twiddle／設定資料，`fft_init()` 之後不會再向 heap 要任何記憶體。

### 逐函式簽章與呼叫時機

```c
FftHandle* fft_create(int fft_size);
size_t     fft_get_mem_size(int fft_size);
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);
void       fft_destroy(FftHandle* handle);
int        fft_get_n_freqs(const FftHandle* handle);

void fft_forward(FftHandle* handle, const float* restrict time_in,
                 Complex* restrict freq_out);
void fft_inverse(FftHandle* handle, const Complex* restrict freq_in,
                 float* restrict time_out);
void fft_forward_scratch(FftHandle* handle, float* time_in_clobbered,
                         Complex* complex_out);
void fft_inverse_scratch(FftHandle* handle, Complex* freq_in_clobbered,
                         float* real_out);

void fft_power(const Complex* freq, float* power, int n_freqs);
void fft_apply_gain(Complex* freq, const float* gain, int n_freqs);
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `fft_create` / `fft_init` | 初始化時一次 | `fft_size` 必須是 2 的冪且落在 `[16, 8192]`。其他值（含其他 2 的冪，如 8 或 16384）一律被拒絕 |
| `fft_get_n_freqs` | 初始化後，用來決定頻域緩衝區長度 | 傳入 handle，回傳 `fft_size/2 + 1` |
| `fft_forward` | 每個 hop，時域 → 頻域 | `time_in` 長度 `fft_size`；呼叫後 `time_in` 內容**不變** |
| `fft_inverse` | 每個 hop，頻域 → 時域 | `freq_in` 長度 `n_freqs`；已含 1/N 正規化，不要再自己除 |
| `fft_forward_scratch` | 輸入緩衝區之後用不到、且想省一次複製時 | 呼叫後 `time_in_clobbered` 內容**未定義** |
| `fft_inverse_scratch` | 同上，頻域側 | 呼叫後 `freq_in_clobbered` 內容**未定義** |
| `fft_power` | 需要 \|X[k]\|² 時 | `n_freqs` 由 `fft_get_n_freqs()` 取得 |
| `fft_apply_gain` | 對頻譜逐 bin 套用實數增益（就地） | `gain` 長度 `n_freqs` |

### 緩衝區大小規則

| 緩衝區 | 元素型別 | 長度 |
|---|---|---|
| 時域輸入／輸出 | `float` | `fft_size` |
| 頻域頻譜 | `Complex` | `n_freqs = fft_size/2 + 1` |
| power / gain | `float` | `n_freqs` |

`Complex` 的定義就是 `{ float r; float i; }`，保證是兩個連續、無填補的 `float`（`simd_kernels.h` 有編譯期斷言保護這個佈局）。因此 `n_freqs` 個 `Complex` 佔 `n_freqs * 2 * sizeof(float)` 位元組。

static pool 的緩衝區只要 16-byte 對齊即可，不需要更大的對齊。

### 就地／別名限制

- `fft_forward` / `fft_inverse` 的輸入與輸出**不可以重疊**（宣告上帶 `restrict`），重疊是未定義行為。
- 需要「輸入緩衝區可被覆寫」的語意時用 `_scratch` 版本；它只是**允許**覆寫，不保證一定覆寫，所以呼叫後絕對不要再讀那個緩衝區。
- `fft_apply_gain` 本來就是就地操作（直接改寫 `freq`）。
- `fft_power` 的輸入與輸出應視為不可重疊。

### 錯誤回傳

- `fft_create` / `fft_init`：失敗回 `NULL`。
- `fft_get_mem_size`：`fft_size` 不合法回 `0`。
- `fft_get_n_freqs`：handle 為 NULL 回 `0`。
- 其餘全為 `void`；傳入 NULL handle 或 NULL 緩衝區時直接 return，不會當機也不會回報。**這代表忘記檢查 `fft_create()` 的回傳值時，後續處理會安靜地什麼都不做，輸出保持原樣。**
- `fft_destroy`：heap handle 只能釋放一次；pool handle 呼叫幾次都安全。

### 常見錯誤

1. 用 `fft_size` 當頻域緩衝區長度（正確是 `n_freqs`），或反過來。
2. 對 `fft_inverse` 的輸出再除一次 N。
3. `fft_size` 填了非 2 的冪（例如 hop 長度 480），`fft_create()` 回 `NULL` 卻沒檢查，之後所有處理安靜失效。
4. 呼叫 `_scratch` 版本後又去讀輸入緩衝區。
5. 期待 `SIMD=1`（NEON）與 `SIMD=0`（純量 C 核心）的 FFT 輸出逐位元相同；兩者只在浮點捨入內一致。
6. static pool 的基底位址沒有 16-byte 對齊，`fft_init()` 回 `NULL`。

---

## audio_resampler.h · 串流重取樣器

### 何時使用

輸入取樣率與處理取樣率不同時。它是有狀態的串流式重取樣器：狀態跨 `process()` 呼叫保留，因此把一段音訊切成任意大小的區塊逐塊送進去，得到的輸出與一次送完整段完全相同。

### 生命週期

**Heap 路徑**

```c
AudioResampler* rs = audio_resampler_create(48000, 16000, 2);
if (!rs) { /* 取樣率或通道數不合法 */ }
/* ... 每個區塊呼叫 audio_resampler_process ... */
audio_resampler_destroy(rs);
```

**Static pool 路徑**

```c
size_t need = audio_resampler_get_mem_size(48000, 16000, 2);
if (need == 0) { /* 參數不合法 */ }
AudioResampler* rs = audio_resampler_init(buf, need, 48000, 16000, 2);  /* buf 需 16-byte 對齊 */
if (!rs) { /* 未對齊 / 空間不足 / 參數不合法 */ }
/* ... 使用 ... */
audio_resampler_destroy(rs);   /* pool 路徑下為 no-op */
```

**物件必須在整段串流期間存活。** 每個區塊重建一次會清掉濾波器歷史，在每個區塊邊界產生 artifact。

### 逐函式簽章與呼叫時機

```c
#define AUDIO_RESAMPLER_MAX_CHANNELS 8

int             audio_resampler_rate_supported(int sample_rate);
AudioResampler* audio_resampler_create(int input_rate, int output_rate, int channels);
size_t          audio_resampler_get_mem_size(int input_rate, int output_rate, int channels);
AudioResampler* audio_resampler_init(void* mem, size_t mem_size,
                                     int input_rate, int output_rate, int channels);
void            audio_resampler_destroy(AudioResampler* self);
void            audio_resampler_reset(AudioResampler* self);

int audio_resampler_process(AudioResampler* self,
                            const float* input, int input_frames,
                            float* output, int output_capacity_frames,
                            int* consumed_frames, int* produced_frames);
int audio_resampler_output_bound(const AudioResampler* self, int input_frames);

int audio_resampler_latency_input_frames(const AudioResampler* self);
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `audio_resampler_rate_supported` | 建立前先驗證來源取樣率 | 允許值只有 `8000`、`16000`、`24000`、`32000`、`48000` |
| `create` / `init` | 串流開始前一次 | 輸入與輸出取樣率各自必須在上述白名單內（可相同）；`channels` 為 `1` 到 `8` |
| `output_bound` | 每個區塊，在 `process()` 之前 | 傳入這次要送的 `input_frames`，回傳輸出緩衝區至少要能裝多少 frame |
| `process` | 每個區塊 | 見下方細節 |
| `reset` | **只在**要丟棄整段串流歷史時（例如換檔案、串流重啟） | 不要在一般區塊邊界呼叫，那會破壞連續性 |
| `latency_input_frames` | 需要做延遲補償對齊時 | 回傳以**輸入 frame** 為單位的群延遲（四捨五入；降取樣比 r 為 `16·r`，升取樣為 16，等取樣率為 0；例：48 kHz → 16 kHz 為 48） |

`process()` 的語意：

- `input` 是交錯（interleaved）的 float32，長度為 `input_frames * channels` 個 float。
- `output` 同樣是交錯的，容量為 `output_capacity_frames * channels` 個 float。
- 成功時回傳 `0`，並且**一定**會寫入 `*consumed_frames` 與 `*produced_frames`。
- 它會盡量處理，直到輸出容量裝不下下一個輸入 frame 可能產生的輸出為止。**若 `*consumed_frames < input_frames`，剩餘輸入沒被消耗，必須在下次呼叫時重送。**
- 輸入與輸出取樣率相同時是精確的 `memmove` 直通，`consumed == produced == min(input_frames, output_capacity_frames)`。

### 緩衝區大小規則

```c
/* 輸入：input_frames * channels 個 float */
/* 輸出：先問，再配置 */
int capacity = audio_resampler_output_bound(rs, input_frames);
if (capacity < 0) { /* handle 為 NULL 或 input_frames 為負，或結果過大 */ }
/* 輸出緩衝區：capacity * channels 個 float */
```

`output_bound()` 給的是保守上界，實際 `produced` 通常會少一點（例如 48 kHz → 16 kHz、480 frames 輸入時，bound 為 161，實際產出 160）。**永遠用 `output_bound()` 的值配置，不要自己用比例推算**，否則在某些相位下會少一個 frame 而導致輸入被截斷。

### 就地／別名限制

- 取樣率**不相同**時，`input` 與 `output` **不可重疊**。
- 取樣率**相同**時是 `memmove` 直通，重疊是安全的。
- 依賴這個差異很脆弱：實務上請一律使用兩個不重疊的緩衝區。

### 錯誤回傳

- `create` / `init`：失敗回 `NULL`（取樣率不在白名單、`channels` 不在 `[1, 8]`、pool 未對齊或太小）。
- `get_mem_size`：參數不合法回 `0`。
- `process`：回 `0` 成功、`-1` 失敗（`self`／`input`／`output`／兩個計數指標任一為 NULL，或 `input_frames`／`output_capacity_frames` 為負）。失敗時**不會**寫入兩個輸出計數。
- `output_bound`：handle 為 NULL、`input_frames` 為負、或結果超出 `int` 範圍時回 `-1`。
- `latency_input_frames`：handle 為 NULL 時回 `-1`。
- `reset` / `destroy`：`void`，NULL 安全。

### 常見錯誤

1. 把 `input_frames` 當成樣本數而不是 frame 數（多通道時緩衝區會少 N 倍）。
2. 不看 `consumed_frames`，以為每次都會吃完，造成靜默丟資料。
3. 輸出緩衝區用「`input_frames * out_rate / in_rate`」推算而不用 `output_bound()`，偶發性少一個 frame。
4. 每個區塊重建 resampler，或在區塊邊界呼叫 `reset()`。
5. 送進非交錯（planar）資料。這個 API 只吃交錯格式。
6. 用了 44100 Hz。它不在白名單內，`create()` 會回 `NULL`。
7. 輸入含 NaN／Inf。一個非有限樣本會污染所有濾波視窗涵蓋它的輸出（NEON 另多讀最多 3 個補零 tap，所以 `SIMD=0` 受污染的輸出略少）；resampler 不檢查，請在送進來前攔截。

---

## hpf.h · 二階高通濾波器

### 何時使用

要移除直流偏移、電源哼聲與低頻隆隆聲時，通常放在處理鏈的最前段（麥克風路徑）。逐樣本、就地處理。

### 生命週期

**Heap 路徑**

```c
Hpf* hpf = hpf_create(80.0f, 16000);
if (!hpf) { /* 參數不在允許定義域 */ }
/* ... 每個區塊呼叫 hpf_process ... */
hpf_destroy(hpf);
```

**Static pool 路徑**

```c
size_t need = hpf_get_mem_size();
Hpf* hpf = hpf_init(buf, need, 80.0f, 16000);   /* buf 需 16-byte 對齊 */
if (!hpf) { /* 未對齊 / 空間不足 / 參數不合法 */ }
/* ... 使用 ... */
hpf_destroy(hpf);        /* pool 路徑下為 no-op */
```

濾波器狀態跨 `hpf_process()` 呼叫保留，物件必須在整段串流期間存活。

### 逐函式簽章與呼叫時機

```c
Hpf*   hpf_create(float cutoff_hz, int sample_rate);
size_t hpf_get_mem_size(void);
Hpf*   hpf_init(void* mem, size_t mem_size, float cutoff_hz, int sample_rate);
void   hpf_destroy(Hpf* hpf);
void   hpf_process(Hpf* hpf, float* data, int n);
void   hpf_reset(Hpf* hpf);
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `hpf_create` / `hpf_init` | 串流開始前一次 | 見下方參數定義域 |
| `hpf_process` | 每個區塊 | `data` 為 mono float32，就地改寫；`n` 為樣本數 |
| `hpf_reset` | 換串流／換檔案時，清除濾波器記憶 | 只清狀態，係數不變 |
| `hpf_destroy` | 結束時 | pool 路徑下為 no-op |

**參數定義域（`hpf_create` 與 `hpf_init` 皆相同）**

- `cutoff_hz` 必須是有限值且 `> 0`。
- `sample_rate` 必須 `> 0`。
- `cutoff_hz` 必須嚴格小於 `0.45 * sample_rate`。

任一條件不滿足（含 `cutoff_hz` 為 NaN 或 Inf）都回 `NULL`，不會產生退化係數。
例：`sample_rate = 16000` 時上限是 `7200`，`hpf_create(7199.0f, 16000)` 成功，`hpf_create(8000.0f, 16000)` 回 `NULL`。

### 緩衝區大小規則

- `data` 是 mono `float` 陣列，長度 `n`。多通道請每個通道各自建立一個 `Hpf` 並各自去交錯後處理。
- static pool 需求由 `hpf_get_mem_size()` 查詢，不帶參數（尺寸與截止頻率無關）。

### 就地／別名限制

`hpf_process()` **只有**就地模式：它直接改寫 `data`。沒有分離輸出的版本。若需要保留原始訊號，請自己先複製一份。

### 錯誤回傳

- `hpf_create` / `hpf_init`：失敗回 `NULL`。`hpf_init` 在基底位址未對齊時**不會寫入該記憶體**就回 `NULL`。
- `hpf_get_mem_size`：僅在尺寸算術溢位時回 `0`（實務上不會發生）。
- `hpf_process` / `hpf_reset` / `hpf_destroy`：`void`。`hpf_process` 在 `hpf` 或 `data` 為 NULL 時安靜 return——**沒濾到任何東西也不會有任何徵兆**。

### 常見錯誤

1. 截止頻率設得太靠近 Nyquist（≥ 0.45×fs），`create()` 回 `NULL` 卻沒檢查，之後 `hpf_process()` 全程 no-op，訊號原封不動通過。
2. 每個區塊重建 `Hpf`，在區塊邊界產生咔聲。
3. 對多通道交錯緩衝區直接呼叫 `hpf_process()`（會把不同通道混進同一組濾波器狀態）。
4. 忘了 `hpf_process()` 是就地的，之後還去讀原始輸入。

---

## mem_align.h · static pool 對齊與尺寸算術

### 何時使用

只有在**你自己也要從同一塊 pool 切出物件**、或要自己計算 pool 總需求時才需要 include 它。
單純呼叫 `X_get_mem_size()` + `X_init()` 的使用者不需要直接用到這個標頭（不過 `ALIGN16()` 在推進 pool 游標時很方便）。

### 生命週期

純巨集與 `static inline` 函式，沒有物件、沒有生命週期、不需要連結封存檔。

### 逐函式／巨集簽章與呼叫時機

```c
#define ALIGN16(x)            /* 把 x 向上取整到 16 的倍數 */
#define MEM_IS_ALIGNED16(p)   /* 指標 p 是否 16-byte 對齊 */
#define MEM_SIZE_INVALID(x)   /* x 是否為溢位哨兵值 SIZE_MAX */

static inline size_t ck_add_size(size_t a, size_t b);
static inline size_t ck_mul_size(size_t a, size_t b);
static inline size_t ck_align16_size(size_t x);
static inline size_t ck_field_size(size_t total, size_t count, size_t elem_size);
```

| 名稱 | 什麼時候用 |
|---|---|
| `ALIGN16(x)` | 在 pool 裡推進游標時：`cursor += ALIGN16(bytes);` |
| `MEM_IS_ALIGNED16(p)` | 自己驗證 pool 基底位址是否合法 |
| `ck_field_size(total, count, elem)` | 累加一個欄位的需求：`total = ck_field_size(total, n, sizeof(float));`。這是尺寸走訪的標準寫法 |
| `ck_field_size_reps(total, count, elem, reps)` | 連續 `reps` 個大小相同、各自對齊 16 的欄位，一次累加 |
| `ck_add_size` / `ck_mul_size` / `ck_align16_size` | 需要更細緻的步驟時 |
| `MEM_SIZE_INVALID(total)` | 走訪結束後檢查是否有任何一步溢位 |

自己寫尺寸查詢時的標準形狀：

```c
size_t my_get_mem_size(int n)
{
    size_t total = 0;
    total = ck_field_size(total, 1, sizeof(MyState));
    total = ck_field_size(total, (size_t)n, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;   /* 溢位時回 0 = 失敗 */
}
```

這些檢查過的算術在任何一步溢位時都會飽和到 `SIZE_MAX` 並一路傳遞下去，所以走訪結尾只要檢查一次 `MEM_SIZE_INVALID()` 就能攔截整條鏈。

### 緩衝區大小規則

- **pool 基底位址必須至少 16-byte 對齊。** 這是整個函式庫共用的唯一對齊要求；所有 `X_init()` 進入點都會檢查，未對齊直接回 `NULL`。
- 尺寸查詢與 init 走訪欄位的順序必須完全一致，回報的大小才會跟實際佈局吻合。這對本函式庫已成立；你自己寫的模組要自己維持。

宣告一塊對齊 pool 的寫法：

```c
static unsigned char g_pool[POOL_BYTES] __attribute__((aligned(16)));
```

### 就地／別名限制

不適用（無資料處理）。

### 錯誤回傳

- `MEM_SIZE_INVALID(total)` 為真代表尺寸算術溢位；此時尺寸查詢**必須**回 `0`（失敗），絕不能回一個被截斷的小數字。
- 巨集本身不回報錯誤。

### 常見錯誤

1. pool 基底沒有對齊到 16 byte（例如 `char pool[N];` 直接用）——所有 `X_init()` 都會回 `NULL`。
2. 推進游標時用 `cursor += bytes` 而不是 `cursor += ALIGN16(bytes)`，導致下一個物件的基底未對齊。
3. 把尺寸查詢回傳的 `0` 當成「不需要記憶體」而繼續往下走。
4. 用一般的 `a * b + c` 算 pool 尺寸而不用 `ck_*`，在極端參數下靜默回繞。
