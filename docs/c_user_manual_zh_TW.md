# audio_common C 使用手冊

本文件是 `audio_common` 共用 DSP 函式庫的**使用者手冊**，對象是「要在自己的 C 專案裡呼叫這個函式庫」的人。
內容只講**什麼時候呼叫哪個函式、參數怎麼填、緩衝區要開多大、錯誤怎麼判斷**；不解釋演算法內部推導或實作歷史。

所有內容以 `include/` 底下的現行標頭檔與 `src/` 現行實作為準；本手冊是這個函式庫使用方式的唯一權威來源。

---

## 1. 這個函式庫是什麼／不是什麼

### 是什麼

`audio_common` 是給 AEC、NR、pipelines 等上層專案共用的**低階 DSP 基礎元件**，一份程式碼、一份封存檔（`.a`），所有消費端共用。它提供：

- 一組 FFT 介面，底下可切換兩種後端實作（KISS 與 NE10），對外 API 完全相同。
- 一個串流式有理數重取樣器（rational polyphase resampler）。
- 一個二階高通濾波器（HPF）與一個 dB 前級增益（pre-gain）。
- 純標頭（header-only）的數學近似函式與 per-bin 微核心，供效能敏感的迴圈直接內嵌使用。
- 一組靜態記憶體配置（static pool）用的對齊與尺寸計算巨集。
- 一個經過強化的 WAV 讀寫器，給測試工具與離線批次程式使用。

### 不是什麼

- **不是**完整的音訊處理管線。它不做回音消除、不做降噪、不做 VAD、不做 AGC，也不管框架（frame）排程。那些在上層專案。
- **不是**通用音訊格式函式庫。WAV 模組只接受兩種取樣格式（見第 5.8 節），其他一律在開檔時就拒絕。
- **不是**執行緒安全的物件庫。每一個 handle（`FftHandle`、`AudioResampler`、`Hpf`、`AudioPreGain`、`WavReader`、`WavWriter`）都帶有可變狀態，**同一個 handle 不可以同時被兩條執行緒使用**。不同 handle 之間彼此獨立，可以平行使用。
- **不是**會替你檢查取樣率一致性的框架。取樣率、通道數、緩衝區長度全部由呼叫端負責對齊。
- **不做**自動裁切（clipping）與自動正規化。pre-gain 明確不做裁切；溢位政策是呼叫端的責任。

### 八個標頭檔的功能地圖

| 標頭檔 | 解決什麼問題 | 型態 | 需要連結 `.a` 嗎 |
|---|---|---|---|
| `include/fft_wrapper.h` | 實數訊號 ↔ 頻譜（rfft / irfft）、頻譜的振幅／功率／增益套用 | opaque handle + 函式 | 是 |
| `include/audio_resampler.h` | 8/16/24/32/48 kHz 之間互轉，可跨區塊連續串流，最多 8 通道交錯 | opaque handle + 函式 | 是 |
| `include/hpf.h` | 去直流、去低頻隆隆聲的二階高通濾波，逐樣本就地處理 | opaque handle + 函式 | 是 |
| `include/audio_pre_gain.h` | 以 dB 設定的輸入增益，逐樣本乘法，支援就地 | opaque handle + 函式 | 是 |
| `include/mem_align.h` | 自己算 static pool 尺寸時的 16-byte 對齊與溢位安全加乘 | 純巨集／`static inline` | 否 |
| `include/simd_kernels.h` | 想把 per-bin 迴圈換成向量化版本（NEON／純量兩套，行為一致） | 純標頭 | 否 |
| `include/fast_math.h` | 熱迴圈裡要 `exp`／`log`／`sqrt`／`E1` 的快速近似版 | 純標頭 | 否 |
| `include/wav_io.h` | 離線工具讀寫 WAV 檔（強化過的解析器 + 兩種寫入風格） | 純標頭 | 否 |

**選用指引（一句話版）**

- 要做頻域處理 → `fft_wrapper.h`
- 輸入取樣率跟處理取樣率不同 → `audio_resampler.h`
- 麥克風訊號有直流偏移／低頻雜訊 → `hpf.h`
- 輸入電平太小或太大 → `audio_pre_gain.h`
- 沒有 heap，要從自備記憶池配置上面這些物件 → `mem_align.h`
- 自己寫的 per-bin 迴圈想加速 → `simd_kernels.h`
- 自己寫的純量迴圈裡有 `expf`／`logf`／`sqrtf` 想加速 → `fast_math.h`
- 要寫離線測試程式讀寫音檔 → `wav_io.h`

---

## 2. 快速上手

以下三段程式碼都已對現行標頭實際編譯並執行過。編譯方式見第 3 節。

### 2.1 FFT 來回轉換

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fft_wrapper.h"

int main(void)
{
    const int fft_size = 512;                 /* 2 的冪，[16, 8192] */

    FftHandle* fft = fft_create(fft_size);
    if (!fft) return 1;                        /* 尺寸不合法或配置失敗 */

    int n_freqs = fft_get_n_freqs(fft);        /* = fft_size/2 + 1 = 257 */

    float*   time_in  = (float*)calloc((size_t)fft_size, sizeof(float));
    float*   time_out = (float*)calloc((size_t)fft_size, sizeof(float));
    Complex* spec     = (Complex*)calloc((size_t)n_freqs, sizeof(Complex));
    float*   power    = (float*)calloc((size_t)n_freqs, sizeof(float));

    for (int i = 0; i < fft_size; ++i)
        time_in[i] = sinf(2.0f * 3.14159265f * 8.0f * (float)i / (float)fft_size);

    fft_forward(fft, time_in, spec);           /* time_in[512] -> spec[257] */
    fft_power(spec, power, n_freqs);           /* 想要功率頻譜時才呼叫 */
    fft_inverse(fft, spec, time_out);          /* spec[257] -> time_out[512] */

    float max_err = 0.0f;
    for (int i = 0; i < fft_size; ++i) {
        float e = fabsf(time_out[i] - time_in[i]);
        if (e > max_err) max_err = e;
    }
    printf("round-trip max error = %.3e\n", (double)max_err);

    free(time_in); free(time_out); free(spec); free(power);
    fft_destroy(fft);
    return 0;
}
```

`fft_inverse()` 已經含 1/N 正規化，來回轉換後直接就是原訊號，呼叫端**不要**再自己除以 N。

### 2.2 重取樣一個區塊

```c
#include <stdio.h>
#include <stdlib.h>
#include "audio_resampler.h"

int main(void)
{
    const int channels  = 2;
    const int in_frames = 480;                 /* frame 數，不是樣本數 */

    AudioResampler* rs = audio_resampler_create(48000, 16000, channels);
    if (!rs) return 1;                          /* 取樣率或通道數不在允許範圍 */

    int capacity = audio_resampler_output_bound(rs, in_frames);
    if (capacity < 0) { audio_resampler_destroy(rs); return 1; }

    float* in  = (float*)calloc((size_t)in_frames * channels, sizeof(float));
    float* out = (float*)calloc((size_t)capacity  * channels, sizeof(float));
    /* ... 把交錯的 float32 輸入填進 in ... */

    int consumed = 0, produced = 0;
    if (audio_resampler_process(rs, in, in_frames, out, capacity,
                                &consumed, &produced) != 0) {
        free(in); free(out); audio_resampler_destroy(rs); return 1;
    }
    printf("consumed=%d produced=%d\n", consumed, produced);

    free(in); free(out);
    audio_resampler_destroy(rs);
    return 0;
}
```

以上參數實測結果：`capacity` 為 161、`consumed` 為 480、`produced` 為 160。
**每個區塊都要檢查 `consumed`**：若 `consumed < input_frames`，代表輸出容量不足，剩下的輸入這次沒被吃掉，下次要重送。

### 2.3 對 mono 緩衝區做 HPF + 前級增益

```c
#include <stdio.h>
#include "hpf.h"
#include "audio_pre_gain.h"

int main(void)
{
    const int sample_rate = 16000;
    const int n           = 160;               /* 10 ms @ 16 kHz */

    Hpf*          hpf  = hpf_create(80.0f, sample_rate);   /* 截止 80 Hz */
    AudioPreGain* gain = audio_pre_gain_create(-6.0f);     /* -6 dB */
    if (!hpf || !gain) return 1;

    float buf[160];
    for (int i = 0; i < n; ++i) buf[i] = 0.5f;  /* 換成你的 mono float32 資料 */

    /* 兩者都支援就地（in-place）：input 與 output 可以是同一個指標 */
    if (audio_pre_gain_process(gain, buf, buf, n) != 0) return 1;
    hpf_process(hpf, buf, n);

    printf("buf[0] = %.6f\n", (double)buf[0]);

    audio_pre_gain_destroy(gain);
    hpf_destroy(hpf);
    return 0;
}
```

`Hpf` 帶有跨呼叫的濾波器狀態，**必須**在整段串流期間存活；每個區塊重新 `hpf_create()` 會在區塊邊界產生不連續。

---

## 3. 建置與連結

### 3.1 建置函式庫

在 `audio_common` 目錄下：

```
make                    # 後端自動偵測：編譯器目標具 ARM NEON -> ne10，否則 kiss
make BACKEND=kiss       # 強制可攜式 KISS 後端
make BACKEND=ne10       # 強制 NE10 後端（ARM NEON）
make SIMD=0             # 後端不變，但把所有可選的 SIMD 路徑改成純量
make selftest           # 往返轉換 + static/heap 位元一致性檢查
make test_audio_utils   # pre-gain 與所有支援的取樣率配對測試
```

`BACKEND` 與 `SIMD` 是公開的建置開關。其他旗標透過 `EXTRA_CFLAGS` 注入。

### 3.2 後端選擇

| 後端 | 什麼時候選 | 特性 |
|---|---|---|
| `kiss` | 桌機／CI／參考建置 | 可攜、可重現，作為對照基準 |
| `ne10` | 目標平台（ARM NEON）交付建置 | NEON FFT 核心 |

**必須刻意選擇，不要靠自動偵測交付。** 自動偵測的結果取決於你的編譯器目標，交叉編譯時可能不是你要的那個。兩個後端數值輸出**不是**逐位元相同（不同 FFT 實作），所以在做位元級回歸比對時，比較的兩邊必須是同一個後端。

同一後端的 heap 路徑與 static pool 路徑輸出必須逐位元相同——配置方式不影響數值。

### 3.3 SIMD 開關

`make SIMD=0` 會對整個封存檔加上 `-DSIMD_KERNELS_FORCE_SCALAR`。這個定義同時影響：

- `simd_kernels.h` 裡所有 `sk_*` 進入點（改為呼叫對應的 `_scalar` 版本，`SK_HAVE_NEON` 變成 0）
- `src/fft_wrapper.c` / `src/fft_wrapper_ne10.c` 裡 `fft_magnitude` / `fft_power` / `fft_apply_gain` 的向量化路徑
- `src/audio_resampler.c` 的向量化內積
- `src/audio_pre_gain.c` 的向量化乘法

**關鍵**：`simd_kernels.h` 與 `fast_math.h` 是純標頭。你自己的 `.c` 檔如果 include 了它們，那份程式碼是編在**你的** translation unit 裡，不是編在封存檔裡。因此封存檔用 `SIMD=0` 建置、而你的 TU 沒有加 `-DSIMD_KERNELS_FORCE_SCALAR`（或反過來），會得到兩種不同的核心行為並存。**兩邊必須一致。**

### 3.4 取得封存檔路徑

輸出路徑同時由後端與完整編譯旗標簽章決定：

```
bin/<backend>-<config-hash>/libaudio_common.a
```

不要自己拼這個路徑。用**與建置時完全相同的旗標**呼叫：

```
make -s BACKEND=kiss print-lib-path      # 只印出路徑，不印其他訊息
make -s BACKEND=ne10 SIMD=0 print-lib-path
```

`print-lib-path` 的標準輸出保證只有路徑一行，可以直接被其他 Makefile 或腳本擷取：

```make
AUDIO_COMMON_LIB := $(shell $(MAKE) -s -C $(AUDIO_COMMON_DIR) BACKEND=$(BACKEND) print-lib-path)
```

若需要穩定的交付路徑，`make publish` 會產生內容定址的釋出目錄並把 `dist/<backend>/current` 指過去。

### 3.5 消費端的編譯旗標

```
-I<audio_common>/include        # 標頭搜尋路徑
-ffp-contract=off               # 強制要求，見下
```

連結時加上 `print-lib-path` 給出的封存檔，並確保有 `-lm`。

**`-ffp-contract=off` 是強制的，而且是對「你自己的 translation unit」的要求。**
`simd_kernels.h` 與 `fast_math.h` 的內容會編進你的 TU。這些核心刻意寫成「分開的乘法再分開的加法」，而編譯器在沒有這個旗標時會自行把 `a*b + c` 融合成 FMA 指令，導致同一份原始碼在你的 TU 與封存檔裡產生不同的位元結果。

- 只要 include 了 `simd_kernels.h` 或 `fast_math.h` 的 TU，一律要加 `-ffp-contract=off`。
- 這個旗標要放在 `CFLAGS` 的**最後**，確保不會被後面的附加項覆寫。
- 不要在任何地方加 `-Ofast`、`-ffast-math` 或其他 `-ffp-contract=` 值；audio_common 的 Makefile 會在解析階段直接拒絕這類 `EXTRA_CFLAGS`。

消費端 TU 用 `-std=c99` 或 `-std=gnu99` 都可以（不同 TU 用不同 std 是合法的），但 `wav_io.h` 需要 `gnu99` 以上的前處理器支援。

編譯本手冊中的範例：

```
cc -std=gnu99 -O2 -Wall -Wextra -ffp-contract=off \
   -I<audio_common>/include your_code.c "$(make -s -C <audio_common> print-lib-path)" -lm -o your_app
```

---

## 4. 錯誤回傳慣例

這個函式庫混用了數種回傳慣例。下表是逐函式的實際行為，**請以本表為準，不要類推**。

| 慣例 | 意義 | 適用函式 |
|---|---|---|
| 回傳 `NULL` | 建構失敗（參數不合法、記憶體不足、pool 太小或未對齊、檔案不合法） | `fft_create`、`fft_init`、`hpf_create`、`hpf_init`、`audio_pre_gain_create`、`audio_pre_gain_init`、`audio_resampler_create`、`audio_resampler_init`、`wav_open_read`、`wav_open_write` |
| 回傳 `0`（`size_t`） | 尺寸查詢失敗；**`0` 永遠代表失敗，不代表「不需要記憶體」** | `fft_get_mem_size`（`fft_size` 不合法）、`audio_resampler_get_mem_size`（取樣率／通道數不合法）、`hpf_get_mem_size`、`audio_pre_gain_get_mem_size`（僅在尺寸算術溢位時，實務上不會發生） |
| 回傳 `0` / `-1`（`int`） | `0` 成功、`-1` 失敗（handle 為 NULL、指標為 NULL、數量為負） | `audio_resampler_process`、`audio_pre_gain_process`、`audio_pre_gain_set_db`、`wav_finalize_write` |
| 回傳 `-1` 當哨兵值 | handle 為 NULL（或參數不合法）時回傳 `-1`，否則回傳真值 | `audio_resampler_input_rate`、`audio_resampler_output_rate`、`audio_resampler_channels`、`audio_resampler_latency_input_frames`、`audio_resampler_output_bound` |
| 回傳 `NaN` | handle 為 NULL 時回傳 `NaN`（要用 `isnan()` 判斷，不能用 `== -1`） | `audio_pre_gain_get_db`、`audio_pre_gain_get_linear` |
| 回傳 `0` 當哨兵值 | handle 為 NULL 時回傳 `0` | `fft_get_n_freqs` |
| 回傳 `1` / `0` 當布林 | `1` = 支援、`0` = 不支援 | `audio_resampler_rate_supported` |
| 回傳實際筆數 | 回傳真正讀到的樣本數；`0` 代表 NULL 參數或已到檔尾 | `wav_read_float` |
| `void`，NULL 時安靜略過 | 傳入 NULL handle／NULL 緩衝區時直接 return，不當機、不回報 | `fft_forward`、`fft_inverse`、`fft_forward_scratch`、`fft_inverse_scratch`、`fft_magnitude`、`fft_power`、`fft_phase`、`fft_from_mag_phase`、`fft_apply_gain`、`hpf_process`、`hpf_reset`、`audio_resampler_reset`、`wav_write_float` |
| `void`，釋放語意 | NULL 安全；static pool 建立的 handle 一律是 no-op | `fft_destroy`、`hpf_destroy`、`audio_pre_gain_destroy`、`audio_resampler_destroy` |
| `void`，吞掉結果 | 內部呼叫 `wav_finalize_write()` 但丟棄其回傳值 | `wav_close_write` |
| `void`，NULL 安全 | 關閉並釋放讀取器 | `wav_close_read` |
| **無錯誤通道** | 不檢查 NULL、不檢查範圍；超出定義域的輸入回傳固定的邊界值（見第 5.7 節） | 所有 `sk_*` 核心、所有 `fast_*` / `clip_f` / `min_f` / `max_f` / `exp1_approx` |

三個最容易寫錯的判斷：

```c
/* 對 */ if (audio_pre_gain_process(g, in, out, n) != 0) { /* 失敗 */ }
/* 對 */ if (isnan(audio_pre_gain_get_db(g)))            { /* handle 是 NULL */ }
/* 對 */ size_t need = fft_get_mem_size(sz); if (need == 0) { /* 尺寸不合法 */ }

/* 錯 */ if (!audio_pre_gain_process(g, in, out, n)) { /* 這是「成功」！ */ }
/* 錯 */ if (audio_pre_gain_get_db(g) == -1.0f)      { /* 永遠不會成立 */ }
/* 錯 */ if (fft_get_mem_size(sz) >= 0)              { /* size_t 永遠 >= 0 */ }
```

---

## 5. 逐模組說明

每個模組都用同一套架構說明：**何時使用 → 生命週期 → 逐函式簽章與呼叫時機 → 緩衝區大小規則 → 就地／別名限制 → 錯誤回傳 → 常見錯誤**。

---

### 5.1 `fft_wrapper.h` — FFT 包裝層

#### 何時使用

需要把時域實數訊號轉到頻域、或把頻域轉回時域時。典型用法是每個 hop 做一次 `fft_forward`，在頻域做處理，再做一次 `fft_inverse`。
兩種後端（KISS / NE10）共用這一組 API，切換後端不需要改任何呼叫端程式碼。

#### 生命週期

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

`fft_get_mem_size()` 回報的位元組數**已包含**後端內部的 twiddle／設定資料，兩個後端在 `fft_init()` 之後都不會再向 heap 要任何記憶體。

#### 逐函式簽章與呼叫時機

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

void fft_magnitude(const Complex* freq, float* magnitude, int n_freqs);
void fft_power(const Complex* freq, float* power, int n_freqs);
void fft_phase(const Complex* freq, float* phase, int n_freqs);
void fft_from_mag_phase(const float* magnitude, const float* phase,
                        Complex* spectrum, int n_freqs);
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
| `fft_magnitude` | 需要 \|X[k]\| 時 | `n_freqs` 由 `fft_get_n_freqs()` 取得 |
| `fft_power` | 需要 \|X[k]\|² 時（比 magnitude 便宜，不開根號） | 同上 |
| `fft_phase` | 需要每個 bin 的相位（弧度）時 | 同上 |
| `fft_from_mag_phase` | 由振幅與相位重建複數頻譜時 | 三個陣列長度都是 `n_freqs` |
| `fft_apply_gain` | 對頻譜逐 bin 套用實數增益（就地） | `gain` 長度 `n_freqs` |

**關於 `fft_phase()` 與 `fft_from_mag_phase()`**：這兩個函式是**受支援的公開 API**。它們在標頭中宣告、在兩個後端都有實作、語意明確（`atan2f(im, re)` 逐 bin；以及 `magnitude * (cos + i·sin)` 逐 bin）、且與其他頻譜輔助函式共用同一套 NULL 保護與長度慣例。目前 audio_common 內部沒有呼叫者，但這不影響它們可被使用。

#### 緩衝區大小規則

| 緩衝區 | 元素型別 | 長度 |
|---|---|---|
| 時域輸入／輸出 | `float` | `fft_size` |
| 頻域頻譜 | `Complex` | `n_freqs = fft_size/2 + 1` |
| magnitude / power / phase / gain | `float` | `n_freqs` |

`Complex` 的定義就是 `{ float r; float i; }`，保證是兩個連續、無填補的 `float`（`simd_kernels.h` 有編譯期斷言保護這個佈局）。因此 `n_freqs` 個 `Complex` 佔 `n_freqs * 2 * sizeof(float)` 位元組。

static pool 的緩衝區只要 16-byte 對齊即可，不需要更大的對齊。

#### 就地／別名限制

- `fft_forward` / `fft_inverse` 的輸入與輸出**不可以重疊**（宣告上帶 `restrict`），重疊是未定義行為。
- 需要「輸入緩衝區可被覆寫」的語意時用 `_scratch` 版本；它只是**允許**覆寫，不保證一定覆寫，所以呼叫後絕對不要再讀那個緩衝區。
- `fft_apply_gain` 本來就是就地操作（直接改寫 `freq`）。
- `fft_magnitude` / `fft_power` / `fft_phase` / `fft_from_mag_phase` 的輸入與輸出應視為不可重疊。

#### 錯誤回傳

- `fft_create` / `fft_init`：失敗回 `NULL`。
- `fft_get_mem_size`：`fft_size` 不合法回 `0`。
- `fft_get_n_freqs`：handle 為 NULL 回 `0`。
- 其餘全為 `void`；傳入 NULL handle 或 NULL 緩衝區時直接 return，不會當機也不會回報。**這代表忘記檢查 `fft_create()` 的回傳值時，後續處理會安靜地什麼都不做，輸出保持原樣。**
- `fft_destroy`：heap handle 只能釋放一次；pool handle 呼叫幾次都安全。

#### 常見錯誤

1. 用 `fft_size` 當頻域緩衝區長度（正確是 `n_freqs`），或反過來。
2. 對 `fft_inverse` 的輸出再除一次 N。
3. `fft_size` 填了非 2 的冪（例如 hop 長度 480），`fft_create()` 回 `NULL` 卻沒檢查，之後所有處理安靜失效。
4. 呼叫 `_scratch` 版本後又去讀輸入緩衝區。
5. 期待 KISS 與 NE10 後端輸出逐位元相同。
6. static pool 的基底位址沒有 16-byte 對齊，`fft_init()` 回 `NULL`。

---

### 5.2 `audio_resampler.h` — 串流重取樣器

#### 何時使用

輸入取樣率與處理取樣率不同時。它是有狀態的串流式重取樣器：狀態跨 `process()` 呼叫保留，因此把一段音訊切成任意大小的區塊逐塊送進去，得到的輸出與一次送完整段完全相同。

#### 生命週期

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

#### 逐函式簽章與呼叫時機

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

int audio_resampler_input_rate(const AudioResampler* self);
int audio_resampler_output_rate(const AudioResampler* self);
int audio_resampler_channels(const AudioResampler* self);
int audio_resampler_latency_input_frames(const AudioResampler* self);
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `audio_resampler_rate_supported` | 建立前先驗證來源取樣率 | 允許值只有 `8000`、`16000`、`24000`、`32000`、`48000` |
| `create` / `init` | 串流開始前一次 | 輸入與輸出取樣率各自必須在上述白名單內（可相同）；`channels` 為 `1` 到 `8` |
| `output_bound` | 每個區塊，在 `process()` 之前 | 傳入這次要送的 `input_frames`，回傳輸出緩衝區至少要能裝多少 frame |
| `process` | 每個區塊 | 見下方細節 |
| `reset` | **只在**要丟棄整段串流歷史時（例如換檔案、串流重啟） | 不要在一般區塊邊界呼叫，那會破壞連續性 |
| `input_rate` / `output_rate` / `channels` | 需要回查設定時 | — |
| `latency_input_frames` | 需要做延遲補償對齊時 | 回傳以**輸入 frame** 為單位的群延遲 |

`process()` 的語意：

- `input` 是交錯（interleaved）的 float32，長度為 `input_frames * channels` 個 float。
- `output` 同樣是交錯的，容量為 `output_capacity_frames * channels` 個 float。
- 成功時回傳 `0`，並且**一定**會寫入 `*consumed_frames` 與 `*produced_frames`。
- 它會盡量處理，直到輸出容量裝不下下一個輸入 frame 可能產生的輸出為止。**若 `*consumed_frames < input_frames`，剩餘輸入沒被消耗，必須在下次呼叫時重送。**
- 輸入與輸出取樣率相同時是精確的 `memmove` 直通，`consumed == produced == min(input_frames, output_capacity_frames)`。

#### 緩衝區大小規則

```c
/* 輸入：input_frames * channels 個 float */
/* 輸出：先問，再配置 */
int capacity = audio_resampler_output_bound(rs, input_frames);
if (capacity < 0) { /* handle 為 NULL 或 input_frames 為負，或結果過大 */ }
/* 輸出緩衝區：capacity * channels 個 float */
```

`output_bound()` 給的是保守上界，實際 `produced` 通常會少一點（例如 48 kHz → 16 kHz、480 frames 輸入時，bound 為 161，實際產出 160）。**永遠用 `output_bound()` 的值配置，不要自己用比例推算**，否則在某些相位下會少一個 frame 而導致輸入被截斷。

#### 就地／別名限制

- 取樣率**不相同**時，`input` 與 `output` **不可重疊**。
- 取樣率**相同**時是 `memmove` 直通，重疊是安全的。
- 依賴這個差異很脆弱：實務上請一律使用兩個不重疊的緩衝區。

#### 錯誤回傳

- `create` / `init`：失敗回 `NULL`（取樣率不在白名單、`channels` 不在 `[1, 8]`、pool 未對齊或太小）。
- `get_mem_size`：參數不合法回 `0`。
- `process`：回 `0` 成功、`-1` 失敗（`self`／`input`／`output`／兩個計數指標任一為 NULL，或 `input_frames`／`output_capacity_frames` 為負）。失敗時**不會**寫入兩個輸出計數。
- `output_bound`：handle 為 NULL、`input_frames` 為負、或結果超出 `int` 範圍時回 `-1`。
- `input_rate` / `output_rate` / `channels` / `latency_input_frames`：handle 為 NULL 時回 `-1`。
- `reset` / `destroy`：`void`，NULL 安全。

#### 常見錯誤

1. 把 `input_frames` 當成樣本數而不是 frame 數（多通道時緩衝區會少 N 倍）。
2. 不看 `consumed_frames`，以為每次都會吃完，造成靜默丟資料。
3. 輸出緩衝區用「`input_frames * out_rate / in_rate`」推算而不用 `output_bound()`，偶發性少一個 frame。
4. 每個區塊重建 resampler，或在區塊邊界呼叫 `reset()`。
5. 送進非交錯（planar）資料。這個 API 只吃交錯格式。
6. 用了 44100 Hz。它不在白名單內，`create()` 會回 `NULL`。

---

### 5.3 `hpf.h` — 二階高通濾波器

#### 何時使用

要移除直流偏移、電源哼聲與低頻隆隆聲時，通常放在處理鏈的最前段（麥克風路徑）。逐樣本、就地處理。

#### 生命週期

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

#### 逐函式簽章與呼叫時機

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

#### 緩衝區大小規則

- `data` 是 mono `float` 陣列，長度 `n`。多通道請每個通道各自建立一個 `Hpf` 並各自去交錯後處理。
- static pool 需求由 `hpf_get_mem_size()` 查詢，不帶參數（尺寸與截止頻率無關）。

#### 就地／別名限制

`hpf_process()` **只有**就地模式：它直接改寫 `data`。沒有分離輸出的版本。若需要保留原始訊號，請自己先複製一份。

#### 錯誤回傳

- `hpf_create` / `hpf_init`：失敗回 `NULL`。`hpf_init` 在基底位址未對齊時**不會寫入該記憶體**就回 `NULL`。
- `hpf_get_mem_size`：僅在尺寸算術溢位時回 `0`（實務上不會發生）。
- `hpf_process` / `hpf_reset` / `hpf_destroy`：`void`。`hpf_process` 在 `hpf` 或 `data` 為 NULL 時安靜 return——**沒濾到任何東西也不會有任何徵兆**。

#### 常見錯誤

1. 截止頻率設得太靠近 Nyquist（≥ 0.45×fs），`create()` 回 `NULL` 卻沒檢查，之後 `hpf_process()` 全程 no-op，訊號原封不動通過。
2. 每個區塊重建 `Hpf`，在區塊邊界產生咔聲。
3. 對多通道交錯緩衝區直接呼叫 `hpf_process()`（會把不同通道混進同一組濾波器狀態）。
4. 忘了 `hpf_process()` 是就地的，之後還去讀原始輸入。

---

### 5.4 `audio_pre_gain.h` — dB 前級增益

#### 何時使用

輸入電平需要整體提升或衰減時，放在處理鏈最前面。dB 值只在 `create` / `init` / `set_db` 時換算一次，`process()` 每個樣本只做一次乘法。

#### 生命週期

**Heap 路徑**

```c
AudioPreGain* g = audio_pre_gain_create(-6.0f);
if (!g) { /* gain_db 不是有限值，或換算結果不是有限值 */ }
/* ... 每個區塊呼叫 audio_pre_gain_process ... */
audio_pre_gain_destroy(g);
```

**Static pool 路徑**

```c
size_t need = audio_pre_gain_get_mem_size();
AudioPreGain* g = audio_pre_gain_init(buf, need, -6.0f);   /* buf 需 16-byte 對齊 */
if (!g) { /* 未對齊 / 空間不足 / gain_db 不合法 */ }
/* ... 使用 ... */
audio_pre_gain_destroy(g);     /* pool 路徑下為 no-op */
```

物件本身沒有跨呼叫的訊號狀態（只存增益值），但仍必須在 `process()` 期間存活。

#### 逐函式簽章與呼叫時機

```c
AudioPreGain* audio_pre_gain_create(float gain_db);
size_t        audio_pre_gain_get_mem_size(void);
AudioPreGain* audio_pre_gain_init(void* mem, size_t mem_size, float gain_db);
void          audio_pre_gain_destroy(AudioPreGain* self);

int   audio_pre_gain_set_db(AudioPreGain* self, float gain_db);
float audio_pre_gain_get_db(const AudioPreGain* self);
float audio_pre_gain_get_linear(const AudioPreGain* self);

int audio_pre_gain_process(const AudioPreGain* self,
                           const float* input, float* output, int n_samples);
```

| 函式 | 什麼時候呼叫 | 參數怎麼填 |
|---|---|---|
| `create` / `init` | 初始化時一次 | `gain_db` 為振幅 dB（`0.0f` = 不變、負值衰減、正值放大），必須是有限值 |
| `set_db` | 執行期要改增益時 | 換算失敗時**不會**動到既有增益 |
| `get_db` | 回查目前設定 | handle 為 NULL 時回 `NaN` |
| `get_linear` | 需要線性倍率（例如記錄 log）時 | handle 為 NULL 時回 `NaN`；例：`-6.0f` dB 對應約 `0.501187` |
| `process` | 每個區塊 | `n_samples` 是樣本數；多通道交錯資料可以直接整段送（增益對所有通道相同） |
| `destroy` | 結束時 | pool 路徑下為 no-op |

**`process()` 不做裁切。** 正增益可能讓樣本超出 `[-1, 1]`。是否要裁切、在哪裡裁切，是呼叫端的決定。

#### 緩衝區大小規則

- `input` 與 `output` 各需 `n_samples` 個 `float`。
- 交錯的多通道資料直接把 `n_samples` 填成 `frames * channels` 即可——這個模組不需要知道通道結構。

#### 就地／別名限制

**支援完全就地**：`input` 與 `output` 可以是**同一個指標**。
不支援部分重疊（`output == input + k`，`k != 0`）。

#### 錯誤回傳

- `create` / `init`：失敗回 `NULL`（`gain_db` 為 NaN／Inf、換算結果非有限、pool 未對齊或太小）。
- `get_mem_size`：僅在算術溢位時回 `0`（實務上不會發生）。
- `set_db`：`0` 成功、`-1` 失敗。
- `get_db` / `get_linear`：handle 為 NULL 時回 **`NaN`**。要用 `isnan()` 判斷。
- `process`：`0` 成功、`-1` 失敗（`self`／`input`／`output` 任一為 NULL，或 `n_samples < 0`）。`n_samples == 0` 是合法的，回 `0`。
- `destroy`：`void`，NULL 安全。

#### 常見錯誤

1. 把 `gain_db` 當成功率 dB。這裡是振幅 dB（`-6 dB` ≈ 0.5 倍振幅）。
2. 用 `== -1.0f` 判斷 `get_db()` 失敗（實際是 `NaN`，必須用 `isnan()`）。
3. 以為 `process()` 會裁切，結果下游收到超過 `[-1, 1]` 的樣本。
4. 多通道時把 `n_samples` 填成 frame 數，只有前面 1/N 的資料被套用增益。

---

### 5.5 `mem_align.h` — static pool 對齊與尺寸算術

#### 何時使用

只有在**你自己也要從同一塊 pool 切出物件**、或要自己計算 pool 總需求時才需要 include 它。
單純呼叫 `X_get_mem_size()` + `X_init()` 的使用者不需要直接用到這個標頭（不過 `ALIGN16()` 在推進 pool 游標時很方便）。

#### 生命週期

純巨集與 `static inline` 函式，沒有物件、沒有生命週期、不需要連結封存檔。

#### 逐函式／巨集簽章與呼叫時機

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

#### 緩衝區大小規則

- **pool 基底位址必須至少 16-byte 對齊。** 這是整個函式庫共用的唯一對齊要求；所有 `X_init()` 進入點都會檢查，未對齊直接回 `NULL`。
- 尺寸查詢與 init 走訪欄位的順序必須完全一致，回報的大小才會跟實際佈局吻合。這對本函式庫已成立；你自己寫的模組要自己維持。

宣告一塊對齊 pool 的寫法：

```c
static unsigned char g_pool[POOL_BYTES] __attribute__((aligned(16)));
```

#### 就地／別名限制

不適用（無資料處理）。

#### 錯誤回傳

- `MEM_SIZE_INVALID(total)` 為真代表尺寸算術溢位；此時尺寸查詢**必須**回 `0`（失敗），絕不能回一個被截斷的小數字。
- 巨集本身不回報錯誤。

#### 常見錯誤

1. pool 基底沒有對齊到 16 byte（例如 `char pool[N];` 直接用）——所有 `X_init()` 都會回 `NULL`。
2. 推進游標時用 `cursor += bytes` 而不是 `cursor += ALIGN16(bytes)`，導致下一個物件的基底未對齊。
3. 把尺寸查詢回傳的 `0` 當成「不需要記憶體」而繼續往下走。
4. 用一般的 `a * b + c` 算 pool 尺寸而不用 `ck_*`，在極端參數下靜默回繞。

---

### 5.6 `simd_kernels.h` — per-bin 微核心（純標頭）

#### 何時使用

當你自己的 per-bin／per-sample 迴圈是熱點、而下表剛好有形狀一致的核心時。
每個核心都有兩個進入點：`sk_<name>()`（在支援的平台上走 NEON）與 `sk_<name>_scalar()`（永遠是純量）。**兩者對任何有限輸入產生逐位元相同的結果**，所以可以在效能與可攜性之間切換而不改變數值輸出。

如果你的迴圈形狀跟下表都不一樣，就不要硬套——直接寫你自己的純量迴圈。

#### 生命週期

純標頭，`static inline`，沒有 handle、沒有初始化、沒有釋放，不需要連結封存檔。
只需要 `-I<audio_common>/include` 加上 `-ffp-contract=off`。

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
void sk_fast_log10_f32(const float *x, float *out, int n);
void sk_exp1_approx_f32(const float *x, float *out, int n);
void sk_mcra_noise_update_f32(float *noise_psd, const float *spp,
                              const float *power,
                              float alpha_d, float bb_scale, int n);
/* 以上每一個都有對應的 sk_<name>_scalar(...)，參數完全相同 */
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
| `sk_fast_log_f32` / `sk_fast_log10_f32` | `fast_log` / `fast_log10` 陣列版 | 大量對數 |
| `sk_exp1_approx_f32` | `exp1_approx` 陣列版 | 大量 E1 近似 |
| `sk_mcra_noise_update_f32` | 每個 bin 各自算權重的 EMA 更新 | 權重隨 bin 變動的雜訊功率更新 |

**選 `sk_*` 還是 `sk_*_scalar`？** 平常一律呼叫不帶後綴的版本，它會自動依建置設定選路。只有在要做「NEON vs 純量」對照驗證時才直接呼叫 `_scalar` 版本。

**數值定義域**：`sk_fast_*` 與 `sk_exp1_approx_f32` 的每個元素行為與 `fast_math.h` 的同名函式完全一致（含定義域邊界值），見第 5.7 節。

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
| `sk_capply_gain_f32` | 支援 `out == z`（完全相同指標） |
| `sk_fast_exp_f32`、`sk_fast_exp_neg_f32`、`sk_fast_log_f32`、`sk_exp1_approx_f32` | 支援 `out == x`（完全相同指標） |
| `sk_mcra_noise_update_f32` | `noise_psd` 是就地讀寫；`spp`／`power` 必須與它不重疊 |
| 其他所有核心（含 `sk_fast_log10_f32`、`sk_fast_sqrt_f32`、`sk_min_f32`、`sk_cadd_f32`、`sk_sq_scale_f32`） | **不支援任何別名** |

**部分重疊（`out == x + k`，`k != 0`）在所有核心一律不支援**，即使某次執行看起來正常。

#### 錯誤回傳

**沒有錯誤通道。** 所有核心都是 `void`，不檢查 NULL 指標、不檢查 `n` 的範圍。傳入 NULL 會直接當機；傳入負的 `n` 會讓迴圈不執行。呼叫端必須自己保證參數正確。

#### 常見錯誤

1. include 了這個標頭但 TU 沒加 `-ffp-contract=off`，導致數值與封存檔內的同名邏輯不一致。
2. 封存檔用 `SIMD=0` 建、自己的 TU 沒定義 `SIMD_KERNELS_FORCE_SCALAR`（或反過來），兩邊走不同路徑。
3. 用了未列在支援表中的別名形式（尤其是 `sk_fast_log10_f32` 的 `out == x`）。
4. 把 `n` 當成位元組數。
5. 對 `Complex` 陣列傳入 `n * 2`（誤把實部虛部各算一個元素）。

---

### 5.7 `fast_math.h` — 快速數學近似（純標頭）

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
static inline float exp1_approx(float v);    /* 指數積分 E1(v) 的三段近似 */

static inline float clip_f(float x, float min_val, float max_val);
static inline float max_f(float a, float b);
static inline float min_f(float a, float b);
```

| 函式 | 有效輸入範圍 | 超出範圍時的行為 |
|---|---|---|
| `fast_exp(x)` | `x ∈ [-16, 16]` | `x < -16` 或 `x` 為 NaN → `0.0f`；`x > 16`（含 `+Inf`）→ `8.8861105e+06f` |
| `fast_exp_neg(x)` | `x ∈ [0, 16]` | `x <= 0` → `1.0f`；`x >= 16` → `0.0f` |
| `fast_log(x)` | `x > 0` 的有限值 | `x <= 0`（含 `±0`）或 NaN → `-1e10f`（代表 −∞） |
| `fast_log10(x)` | 同 `fast_log` | 同 `fast_log`，再乘上 `FM_LOG10E` |
| `fast_sqrt(v)` | `v > 0` 的有限值 | `v <= 0`（含 `-0.0f`）或 NaN → `0.0f`（正零） |
| `exp1_approx(v)` | `v > 0` | `v <= FM_EPSILON` 會先被夾成 `FM_EPSILON` 再計算 |
| `clip_f` / `max_f` / `min_f` | 任意有限值 | 無特殊處理，就是單純的比較選擇 |

**重點：這些函式對非有限輸入回傳的是「定義域邊界常數」，不是 IEEE 語意的結果。**
例如 `fast_sqrt(NaN)` 回 `0.0f`（不是 NaN）、`fast_log(-1.0f)` 回 `-1e10f`（不是 NaN）、`fast_exp(+Inf)` 回一個有限的飽和值（不是 `+Inf`）。這些值是確定且穩定的，但**不具數學意義**。正確做法是在資料進入 DSP 之前就把非有限樣本清掉（`wav_read_float()` 在讀檔端已經做了這件事），而不是依賴這些邊界值。

#### 使用方式

```c
#include "fast_math.h"

for (int k = 0; k < n_freqs; ++k) {
    float mag = fast_sqrt(power[k]);
    float db  = 20.0f * fast_log10(mag + FM_EPSILON);   /* 先加 epsilon 避開 0 */
    gain[k]   = clip_f(gain[k], 0.05f, 1.0f);
}
```

#### 緩衝區大小規則

不適用——全部都是純量對純量的函式。要對整個陣列做同樣運算時，請改用 `simd_kernels.h` 的 `sk_fast_*` 陣列版本（第 5.6 節），它們的每元素行為與這裡完全一致。

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

### 5.8 `wav_io.h` — WAV 讀寫（純標頭）

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
| `wav_close_write` | 舊有程式碼相容用 | 等同 `wav_finalize_write()` 但丟棄回傳值 |

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
5. 餵 24-bit 或 44.1 kHz 以外沒問題但格式不合的檔案，`wav_open_read()` 回 `NULL` 卻沒檢查。
6. 在即時路徑上使用這個模組。

---

## 6. 整合檢查清單

在把 `audio_common` 併入專案、以及每次調整建置設定後，逐項確認：

**取樣率與參數**

- [ ] 送進 `audio_resampler` 的輸入與輸出取樣率都在 `{8000, 16000, 24000, 32000, 48000}` 之內；不確定時先用 `audio_resampler_rate_supported()` 檢查。
- [ ] `channels` 在 `[1, 8]` 之內（`AUDIO_RESAMPLER_MAX_CHANNELS`）。
- [ ] `fft_size` 是 2 的冪且在 `[16, 8192]` 之內。
- [ ] HPF 的 `cutoff_hz` 嚴格小於 `0.45 * sample_rate`，且為有限正值。
- [ ] 讀入的 WAV 只會是 PCM16 或 float32、通道數 `[1, 8]`、取樣率 `[1, 384000]`。

**Static pool**

- [ ] 每一塊 pool 的基底位址都是 16-byte 對齊（例如 `__attribute__((aligned(16)))`）。
- [ ] 每次推進 pool 游標都用 `ALIGN16()`，不是裸的位元組數。
- [ ] 每個 `X_get_mem_size()` 的回傳值都檢查過 `!= 0` 才拿去用。
- [ ] 每個 `X_init()` 的回傳值都檢查過 `!= NULL`。
- [ ] pool 路徑上沒有對同一個 handle 呼叫 `free()`；只用 `X_destroy()`（在 pool 路徑上是 no-op）。

**編譯旗標**

- [ ] 所有 include 了 `simd_kernels.h` 或 `fast_math.h` 的**消費端 TU** 都帶 `-ffp-contract=off`。
- [ ] 該旗標放在 `CFLAGS` 最後，沒有被後續附加項覆寫。
- [ ] 專案中沒有任何地方出現 `-Ofast`、`-ffast-math` 或其他 `-ffp-contract=` 值。
- [ ] 消費端有 `-I<audio_common>/include`，連結時有 `-lm`。

**SIMD 設定一致性**

- [ ] 封存檔的 `SIMD` 設定，與消費端純標頭 TU 的 `SIMD_KERNELS_FORCE_SCALAR` 定義狀態一致（`SIMD=0` ↔ 消費端有定義；`SIMD=1` ↔ 消費端沒定義）。
- [ ] 這個一致性在每個建置設定（debug／release／目標平台／主機）各自成立。
- [ ] 需要時可在執行期印出 `SK_HAVE_NEON` 確認實際走的是哪條路徑。

**後端選擇**

- [ ] 後端是**明確指定**的（`BACKEND=kiss` 或 `BACKEND=ne10`），不是靠自動偵測。
- [ ] 封存檔路徑是用**相同旗標**的 `make -s ... print-lib-path` 取得，不是手寫的。
- [ ] 位元級回歸比對的兩邊使用同一個後端（兩後端輸出不相同）。
- [ ] 交叉編譯時 `CC`／`CXX`／`AR`／`RANLIB` 都已指定，且後端選擇符合目標平台。

**多通道與去交錯**

- [ ] `audio_resampler` 的輸入與輸出都是**交錯**格式，`input_frames` 是 frame 數而非樣本數。
- [ ] 每個區塊都檢查 `consumed_frames`，未消耗的輸入有被重送。
- [ ] 輸出緩衝區大小取自 `audio_resampler_output_bound()`，不是自己按比例算的。
- [ ] `hpf` 是**單通道**模組：多通道時每個通道各有一個 `Hpf` 實例，且已先去交錯。
- [ ] `audio_pre_gain` 用在交錯資料時，`n_samples` 填的是 `frames * channels`。
- [ ] 清楚知道 `wav_read_float()` 只回第一個通道；多通道來源已在上游處理。
- [ ] 不相等取樣率時，resampler 的輸入與輸出緩衝區不重疊。

**生命週期**

- [ ] 所有有狀態的物件（`Hpf`、`AudioResampler`、`FftHandle`）在整段串流期間存活，不是每個區塊重建。
- [ ] `audio_resampler_reset()` 只在真的要丟棄串流歷史時呼叫。
- [ ] 每個 handle 不會被兩條執行緒同時使用。
- [ ] `wav_finalize_write()` 對每個 writer 只呼叫一次，之後不再碰該指標。
