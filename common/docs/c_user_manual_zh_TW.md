# common C 使用手冊

本文件是 `common` 共用 DSP 函式庫的**使用者手冊**，對象是「要在自己的 C 專案裡呼叫這個函式庫」的人。
內容只講**什麼時候呼叫哪個函式、參數怎麼填、緩衝區要開多大、錯誤怎麼判斷**；不解釋演算法內部推導或實作歷史。

所有內容以 `include/` 底下的現行標頭檔與 `src/` 現行實作為準；本手冊是這個函式庫使用方式的唯一權威來源。

HTML 文件站（`audiolib/common/docs/html/`）補上本手冊沒有的視角：呼叫順序與流程時序圖（`audio_common.html`）、資料流／I/O／狀態大小／延遲矩陣（`audio_common_detail.html`）、整合者視角的 API 摘要（`audio_common_c_api.html`），以及哪些參數可以對外開放、哪些只該留在內部、哪些函式該呼叫（`audio_common_params.html`）。同一個 repo 的 `VAD/`、`doa/`、`bf/` 不屬於 common，文件在 `audiolib/docs/html/`。

---

- [common C 使用手冊](#common-c-使用手冊) <!-- split-toc -->
  - [1. 這個函式庫是什麼／不是什麼](#1-這個函式庫是什麼不是什麼) <!-- split-toc -->
    - [是什麼](#是什麼) <!-- split-toc -->
    - [不是什麼](#不是什麼) <!-- split-toc -->
    - [標頭檔的功能地圖](#標頭檔的功能地圖) <!-- split-toc -->
  - [2. 快速上手](#2-快速上手) <!-- split-toc -->
    - [2.1 FFT 來回轉換](#21-fft-來回轉換) <!-- split-toc -->
    - [2.2 重取樣一個區塊](#22-重取樣一個區塊) <!-- split-toc -->
    - [2.3 對 mono 緩衝區做 HPF](#23-對-mono-緩衝區做-hpf) <!-- split-toc -->
  - [3. 建置與連結](#3-建置與連結) <!-- split-toc -->
    - [3.1 建置函式庫](#31-建置函式庫) <!-- split-toc -->
    - [3.2 FFT 後端](#32-fft-後端) <!-- split-toc -->
    - [3.3 SIMD 開關](#33-simd-開關) <!-- split-toc -->
    - [3.4 取得封存檔路徑](#34-取得封存檔路徑) <!-- split-toc -->
    - [3.5 消費端的編譯旗標](#35-消費端的編譯旗標) <!-- split-toc -->
  - [4. 錯誤回傳慣例](#4-錯誤回傳慣例) <!-- split-toc -->
  - [5. 逐模組說明](c_user_manual_zh_TW_01_modules_core.md#5-逐模組說明) <!-- split-toc -->
    - [5.1 `fft_wrapper.h` — FFT 包裝層](c_user_manual_zh_TW_01_modules_core.md#51-fft_wrapperh--fft-包裝層) <!-- split-toc -->
    - [5.2 `audio_resampler.h` — 串流重取樣器](c_user_manual_zh_TW_01_modules_core.md#52-audio_resamplerh--串流重取樣器) <!-- split-toc -->
    - [5.3 `hpf.h` — 二階高通濾波器](c_user_manual_zh_TW_01_modules_core.md#53-hpfh--二階高通濾波器) <!-- split-toc -->
    - [5.4 `mem_align.h` — static pool 對齊與尺寸算術](c_user_manual_zh_TW_01_modules_core.md#54-mem_alignh--static-pool-對齊與尺寸算術) <!-- split-toc -->
    - [5.5 `simd_kernels.h` — per-bin 微核心（純標頭）](c_user_manual_zh_TW_02_modules_util_io.md#55-simd_kernelsh--per-bin-微核心純標頭) <!-- split-toc -->
    - [5.6 `fast_math.h` — 快速數學近似（純標頭）](c_user_manual_zh_TW_02_modules_util_io.md#56-fast_mathh--快速數學近似純標頭) <!-- split-toc -->
    - [5.7 `wav_io.h` — WAV 讀寫（純標頭）](c_user_manual_zh_TW_02_modules_util_io.md#57-wav_ioh--wav-讀寫純標頭) <!-- split-toc -->
    - [5.8 `audio_stft.h` — sqrt-Hann STFT／ISTFT](c_user_manual_zh_TW_02_modules_util_io.md#58-audio_stfth--sqrt-hann-stftistft) <!-- split-toc -->
  - [6. 整合檢查清單](c_user_manual_zh_TW_03_checklist.md#6-整合檢查清單) <!-- split-toc -->
## 1. 這個函式庫是什麼／不是什麼

### 是什麼

`common` 是給 AEC、NR、pipelines 等上層專案共用的**低階 DSP 基礎元件**，一份程式碼、一份封存檔（`.a`），所有消費端共用。它提供：

- 一組 FFT 介面，底層為 NE10（依 ARM32／ARM64 編譯目標自動選路）。
- 一對 sqrt-Hann STFT／ISTFT 物件（50% overlap 組幀、加窗、overlap-add）與週期 sqrt-Hann 窗表。
- 一個串流式有理數重取樣器（rational polyphase resampler）。
- 一個二階高通濾波器（HPF）。
- 純標頭（header-only）的數學近似函式與 per-bin 微核心，供效能敏感的迴圈直接內嵌使用。
- 一組靜態記憶體配置（static pool）用的對齊與尺寸計算巨集。
- 一個經過強化的 WAV 讀寫器，給測試工具與離線批次程式使用。

### 不是什麼

- **不是**完整的音訊處理管線。它不做回音消除、不做降噪、不做 AGC，也不管框架（frame）排程；回音消除與降噪在 AEC、NR 兩個 repo。同一個 audiolib repo 的 `VAD/`、`doa/`、`bf/` 也不屬於 common（見 `audiolib/docs/html/spatial_chain.html`）。
- **不是**通用音訊格式函式庫。WAV 模組只接受兩種取樣格式（見[第 5.7 節](c_user_manual_zh_TW_02_modules_util_io.md#57-wav_ioh--wav-讀寫純標頭)），其他一律在開檔時就拒絕。
- **不是**執行緒安全的物件庫。每一個 handle（`FftHandle`、`AcStft`、`AcIstft`、`AudioResampler`、`Hpf`、`WavReader`、`WavWriter`）都帶有可變狀態，**同一個 handle 不可以同時被兩條執行緒使用**。不同 handle 之間彼此獨立，可以平行使用。
- **不是**會替你檢查取樣率一致性的框架。取樣率、通道數、緩衝區長度全部由呼叫端負責對齊。
- **不做**自動裁切（clipping）與自動正規化；溢位政策是呼叫端的責任。

### 標頭檔的功能地圖

| 標頭檔 | 解決什麼問題 | 型態 | 需要連結 `.a` 嗎 |
|---|---|---|---|
| `include/fft_wrapper.h` | 實數訊號 ↔ 頻譜（rfft / irfft）、頻譜的功率／增益套用 | opaque handle + 函式 | 是 |
| `include/ac_complex.h` | `Complex`（`{ float r; float i; }`）型別，`fft_wrapper.h` 與 `simd_kernels.h` 都 include 它 | 型別 | 否 |
| `include/audio_window.h` | 建立 periodic sqrt-Hann float32 窗表，不依賴 FFT 或組幀物件 | 函式 | 是 |
| `include/audio_stft.h` | sqrt-Hann STFT／ISTFT（50% overlap 組幀、加窗、overlap-add；centered reflect 分析與包絡正規化合成） | opaque handle + 函式 | 是 |
| `include/audio_resampler.h` | 8/16/24/32/48 kHz 之間互轉，可跨區塊連續串流，最多 8 通道交錯 | opaque handle + 函式 | 是 |
| `include/hpf.h` | 去直流、去低頻隆隆聲的二階高通濾波，逐樣本就地處理 | opaque handle + 函式 | 是 |
| `utility/mem_align.h` | 自己算 static pool 尺寸時的 16-byte 對齊與溢位安全加乘 | 純巨集／`static inline` | 否 |
| `utility/simd_kernels.h` | 想把 per-bin 迴圈換成向量化版本（NEON／純量兩套，行為一致） | 純標頭 | 否 |
| `utility/simd_kernel_nn.h` | 神經網路模型前後處理（DFN2／ULCNet／GTCRN／DeepVQE）與 `audio_stft.h` 用的向量化核心，與 `simd_kernels.h` 分開管理 | 純標頭 | 否 |
| `utility/fast_math.h` | 熱迴圈裡要 `exp`／`log`／`sqrt`／`E1` 的快速近似版 | 純標頭 | 否 |
| `utility/simd_aec.h`、`utility/simd_nr.h` | AEC、NR 各自專用的向量化核心本體（整合者不呼叫，由各自 repo 測試） | 純標頭 | 否 |
| `include/wav_io.h` | 離線工具讀寫 WAV 檔（強化過的解析器 + 兩種寫入風格） | 純標頭 | 否 |

**選用指引（一句話版）**

- 要做頻域處理 → `fft_wrapper.h`
- 只需建立 periodic sqrt-Hann 窗表 → `audio_window.h`
- 要做 sqrt-Hann STFT／ISTFT 組幀 → `audio_stft.h`
- 輸入取樣率跟處理取樣率不同 → `audio_resampler.h`
- 麥克風訊號有直流偏移／低頻雜訊 → `hpf.h`
- 沒有 heap，要從自備記憶池配置上面這些物件 → `mem_align.h`
- 自己寫的 per-bin 迴圈想加速 → `simd_kernels.h`
- 自己寫的純量迴圈裡有 `expf`／`logf`／`sqrtf` 想加速 → `fast_math.h`
- 要寫離線測試程式讀寫音檔 → `wav_io.h`

---

## 2. 快速上手

以下三段程式碼可直接編譯執行。編譯方式見第 3 節。

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

以上參數下：`capacity` 為 161、`consumed` 為 480、`produced` 為 160。
**每個區塊都要檢查 `consumed`**：若 `consumed < input_frames`，代表輸出容量不足，剩下的輸入這次沒被吃掉，下次要重送。

### 2.3 對 mono 緩衝區做 HPF

```c
#include <stdio.h>
#include "hpf.h"

int main(void)
{
    const int sample_rate = 16000;
    const int n           = 160;               /* 10 ms @ 16 kHz */

    Hpf* hpf = hpf_create(80.0f, sample_rate);   /* 截止 80 Hz */
    if (!hpf) return 1;

    float buf[160];
    for (int i = 0; i < n; ++i) buf[i] = 0.5f;  /* 換成你的 mono float32 資料 */

    /* 支援就地（in-place）：input 與 output 可以是同一個指標 */
    hpf_process(hpf, buf, n);

    printf("buf[0] = %.6f\n", (double)buf[0]);

    hpf_destroy(hpf);
    return 0;
}
```

`Hpf` 帶有跨呼叫的濾波器狀態，**必須**在整段串流期間存活；每個區塊重新 `hpf_create()` 會在區塊邊界產生不連續。

---

## 3. 建置與連結

### 3.1 建置函式庫

在 `common` 目錄下：

```
make                    # 建置封存檔（NE10 FFT 後端）
make SIMD=0             # 把所有可選的 SIMD 路徑改成純量（NE10 純量 C 核心）
make selftest           # 往返轉換 + static/heap 位元一致性檢查
make test_audio_utils   # 所有支援的取樣率配對測試
make test               # 預設測試層（selftest、test_pool、test_wav、test_wav_nr_style、test_zero_heap、test-fast-sqrt-codegen，加建置隔離套件的 core 情境）
make test-full          # 預設層另加 NEON-vs-C FFT 容差比對與完整建置隔離套件
```

`SIMD` 是公開的建置開關。其他旗標透過 `EXTRA_CFLAGS` 注入。

### 3.2 FFT 後端

NE10 是唯一的 FFT 後端。`BACKEND=ne10` 是預設，也是唯一接受的值；給其他值建置會直接報錯停止。

NE10 包裝層依 ARM32／ARM64 編譯器巨集自動選路；`SIMD=1` 且目標有 `__ARM_NEON` 時使用 NE10 NEON FFT，否則使用同一後端的 C FFT，`SIMD=0` 一律使用 C。共用 DSP／spatial 的 NEON 微核心需要 AArch64 指令，因此 ARM32 保留純量 fallback；ARM32 的 sqrt 使用正確捨入 `sqrtf`（VFP 為 VSQRT，soft-float 為 libm）。非 ARM 主機（例如 x86_64）仍不支援。交叉編譯時把 `CC`／`CXX`／`AR`／`RANLIB` 一起指定為目標工具鏈；ARM32 支援 NEON 時可加 `EXTRA_CFLAGS=-mfpu=neon-vfpv4`（AArch32 ARMv8 核心用 `-mfpu=neon-fp-armv8`）；不要用單純的 `-mfpu=neon`，它沒有硬體 FMA，所有明確的 `fmaf()`（`simd_aec.h` 的 PBFDKF 梯度與權重更新、`simd_kernels.h` 的融合複數乘加 `sk_cmac_fma_ring_f32`、`fft_wrapper_ne10.c` 的功率譜）都會變成 libm 呼叫，位元結果不變但速度慢很多。ARMv7 Advanced SIMD 會把 subnormal 沖為零，並以預設 NaN 取代帶 payload 的 NaN，因此 ARM32 上 `SIMD=1` 與 `SIMD=0` 對 subnormal／NaN payload 的值可能不同；只有 NE10 NEON FFT，以及具硬體 FMA 的核心（`-mfpu=neon-vfpv4`）上 AEC 的 delay matched filter 受影響，其餘 kernel 在 ARM32 皆為 C。不同 ABI 要重新查詢 pool 尺寸／descriptor，不可沿用另一 ABI 的配置。

heap 路徑與 static pool 路徑輸出必須逐位元相同——配置方式不影響數值。NEON 與 `SIMD=0`（NE10 純量 C 核心）兩條路徑只在浮點捨入內一致，不保證逐位元相同（`make test_ne10_force_c` 以容差比對），所以位元級回歸比對的兩邊必須是同一個 `SIMD` 設定。

### 3.3 SIMD 開關

`make SIMD=0` 會對整個封存檔加上 `-DSIMD_KERNELS_FORCE_SCALAR`。這個定義同時影響：

- `simd_kernels.h` 裡所有 `sk_*` 進入點（改為呼叫對應的 `_scalar` 版本，`SK_HAVE_NEON` 變成 0）
- `simd_kernel_nn.h` 裡所有 `skn_*` 進入點（同上，改走 `_scalar` 版本）
- `src/fft_wrapper_ne10.c` 裡 `fft_power` / `fft_apply_gain` 的向量化路徑
- `src/audio_resampler.c` 的向量化內積

**關鍵**：`simd_kernels.h` 與 `fast_math.h` 是純標頭。你自己的 `.c` 檔如果 include 了它們，那份程式碼是編在**你的** translation unit 裡，不是編在封存檔裡。因此封存檔用 `SIMD=0` 建置、而你的 TU 沒有加 `-DSIMD_KERNELS_FORCE_SCALAR`（或反過來），會得到兩種不同的核心行為並存。**兩邊必須一致。**

### 3.4 取得封存檔路徑

輸出路徑由完整編譯旗標簽章決定：

```
bin/ne10-<config-hash>/libaudio_common.a
```

不要自己拼這個路徑。用**與建置時完全相同的旗標**呼叫：

```
make -s print-lib-path                    # 只印出路徑，不印其他訊息
make -s SIMD=0 print-lib-path
```

`print-lib-path` 的標準輸出保證只有路徑一行，可以直接被其他 Makefile 或腳本擷取：

```make
AUDIO_COMMON_LIB := $(shell $(MAKE) -s -C $(AUDIO_COMMON_DIR) print-lib-path)
```

若需要穩定的交付路徑，`make publish` 會產生內容定址的釋出目錄並把 `dist/ne10/current` 指過去。

### 3.5 消費端的編譯旗標

```
-I<common>/include        # 標頭搜尋路徑（頂層標頭）
-I<common>                # 供 utility/... 標頭（#include "utility/fast_math.h"）使用
-ffp-contract=off               # 強制要求，見下
-fno-math-errno                 # AArch64 sqrtf 直接降成 FSQRT，不保留 errno fallback
```

連結時加上 `print-lib-path` 給出的封存檔，並確保有 `-lm`。

**`-ffp-contract=off` 是強制的，而且是對「你自己的 translation unit」的要求。**
`simd_kernels.h` 與 `fast_math.h` 的內容會編進你的 TU。這些核心刻意寫成「分開的乘法再分開的加法」，而編譯器在沒有這個旗標時會自行把 `a*b + c` 融合成 FMA 指令，導致同一份原始碼在你的 TU 與封存檔裡產生不同的位元結果。

- 只要 include 了 `simd_kernels.h` 或 `fast_math.h` 的 TU，一律要加 `-ffp-contract=off -fno-math-errno`。
- 這兩個旗標要放在 `CFLAGS` 的**最後**，確保不會被後面的附加項覆寫。
- 不要在任何地方加 `-Ofast`、`-ffast-math` 或其他 `-ffp-contract=` 值；common 的 Makefile 會在解析階段直接拒絕這類 `EXTRA_CFLAGS`。

消費端 TU 用 `-std=c99` 或 `-std=gnu99` 都可以（不同 TU 用不同 std 是合法的），但 `wav_io.h` 需要 `gnu99` 以上的前處理器支援。

編譯本手冊中的範例：

```
cc -std=gnu99 -O2 -Wall -Wextra -ffp-contract=off -fno-math-errno \
   -I<common>/include -I<common> your_code.c "$(make -s -C <common> print-lib-path)" -lm -o your_app
```

---

## 4. 錯誤回傳慣例

這個函式庫混用了數種回傳慣例。下表是逐函式的實際行為，**請以本表為準，不要類推**。

| 慣例 | 意義 | 適用函式 |
|---|---|---|
| 回傳 `NULL` | 建構失敗（參數不合法、記憶體不足、pool 太小或未對齊、檔案不合法） | `fft_create`、`fft_init`、`ac_stft_init`、`ac_istft_init`、`hpf_create`、`hpf_init`、`audio_resampler_create`、`audio_resampler_init`、`wav_open_read`、`wav_open_write` |
| 回傳 `0`（`size_t`） | 尺寸查詢失敗；**`0` 永遠代表失敗，不代表「不需要記憶體」** | `fft_get_mem_size`（`fft_size` 不合法）、`ac_stft_get_mem_size`／`ac_istft_get_mem_size`（設定不合法）、`audio_resampler_get_mem_size`（取樣率／通道數不合法）、`hpf_get_mem_size` |
| 回傳 `0` / `-1`（`int`） | `0` 成功、`-1` 失敗（handle 為 NULL、指標為 NULL、數量為負） | `audio_resampler_process`、`wav_finalize_write` |
| 回傳 `-1` 當哨兵值 | handle 為 NULL（或參數不合法）時回傳 `-1`，否則回傳真值 | `audio_resampler_latency_input_frames`、`audio_resampler_output_bound` |
| 回傳 `0` 當哨兵值 | handle 為 NULL 時回傳 `0` | `fft_get_n_freqs` |
| 回傳 `1` / `0` 當布林 | `1` = 支援、`0` = 不支援 | `audio_resampler_rate_supported` |
| 回傳實際筆數 | 回傳真正讀到的樣本數；`0` 代表 NULL 參數或已到檔尾 | `wav_read_float` |
| 回傳幀數／樣本數，`0` 兼作 NULL 引數 | `ac_stft_push`／`_planar` 回傳寫出的幀數；`ac_istft_push`／`_planar` 回傳寫出的樣本數（`hop`，`AC_OLA_ENV_NORM` 第 1 幀為 `0`）；`ac_stft_flush_planar`、`ac_istft_flush` 在條件不符時回 `0` | `ac_stft_push`、`ac_stft_push_planar`、`ac_stft_flush_planar`、`ac_istft_push`、`ac_istft_push_planar`、`ac_istft_flush` |
| 回傳 `0` / `-1`（取回設定） | `0` 成功、`-1` 為 NULL 引數 | `ac_stft_get_config`、`ac_istft_get_config` |
| `void`，NULL 時安靜略過 | 傳入 NULL handle／NULL 緩衝區時直接 return，不當機、不回報 | `fft_forward`、`fft_inverse`、`fft_forward_scratch`、`fft_inverse_scratch`、`fft_power`、`fft_apply_gain`、`ac_stft_frame_planar`、`ac_stft_reset`、`ac_istft_reset`、`hpf_process`、`hpf_reset`、`audio_resampler_reset`、`wav_write_float` |
| `void`，釋放語意 | NULL 安全；static pool 建立的 handle 一律是 no-op | `fft_destroy`、`hpf_destroy`、`audio_resampler_destroy` |
| `void`，吞掉結果 | 內部呼叫 `wav_finalize_write()` 但丟棄其回傳值 | `wav_close_write` |
| `void`，NULL 安全 | 關閉並釋放讀取器 | `wav_close_read` |
| **無錯誤通道** | 不檢查 NULL、不檢查範圍；超出定義域的輸入回傳固定的邊界值（見[第 5.6 節](c_user_manual_zh_TW_02_modules_util_io.md#56-fast_mathh--快速數學近似純標頭)） | 所有 `sk_*` 核心、所有 `fast_*` / `clip_f` / `max_f` / `exp1_approx` |

兩個最容易寫錯的判斷：

```c
/* 對 */ if (audio_resampler_process(rs, in, n, out, cap, &used, &made) != 0) { /* 失敗 */ }
/* 對 */ size_t need = fft_get_mem_size(sz); if (need == 0) { /* 尺寸不合法 */ }

/* 錯 */ if (!audio_resampler_process(rs, in, n, out, cap, &used, &made)) { /* 這是「成功」！ */ }
/* 錯 */ if (fft_get_mem_size(sz) >= 0)              { /* size_t 永遠 >= 0 */ }
```

---

