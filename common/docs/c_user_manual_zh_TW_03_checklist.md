> [索引](c_user_manual_zh_TW.md) | [上一部分](c_user_manual_zh_TW_02_modules_util_io.md) <!-- split-nav -->
## 6. 整合檢查清單

在把 `common` 併入專案、以及每次調整建置設定後，逐項確認：

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

- [ ] 所有 include 了 `simd_kernels.h` 或 `fast_math.h` 的**消費端 TU** 都帶 `-ffp-contract=off -fno-math-errno`。
- [ ] 這兩個旗標放在 `CFLAGS` 最後，沒有被後續附加項覆寫。
- [ ] 專案中沒有任何地方出現 `-Ofast`、`-ffast-math` 或其他 `-ffp-contract=` 值。
- [ ] 消費端有 `-I<common>/include` 與 `-I<common>`（`utility/...` 標頭用），連結時有 `-lm`。

**SIMD 設定一致性**

- [ ] 封存檔的 `SIMD` 設定，與消費端純標頭 TU 的 `SIMD_KERNELS_FORCE_SCALAR` 定義狀態一致（`SIMD=0` ↔ 消費端有定義；`SIMD=1` ↔ 消費端沒定義）。
- [ ] 這個一致性在每個建置設定（debug／release／目標平台／主機）各自成立。
- [ ] 需要時可在執行期印出 `SK_HAVE_NEON` 確認實際走的是哪條路徑。

**建置與工具鏈**

- [ ] 目標編譯器為 ARM32／ARM64；依編譯期 NEON 與 `SIMD` 自動選 FFT 路徑，NE10 是唯一後端。
- [ ] 封存檔路徑是用**相同旗標**的 `make -s ... print-lib-path` 取得，不是手寫的。
- [ ] 位元級回歸比對的兩邊使用同一個 `SIMD` 設定。
- [ ] 交叉編譯時 `CC`／`CXX`／`AR`／`RANLIB` 都已指定為同一套目標工具鏈。

**多通道與去交錯**

- [ ] `audio_resampler` 的輸入與輸出都是**交錯**格式，`input_frames` 是 frame 數而非樣本數。
- [ ] 每個區塊都檢查 `consumed_frames`，未消耗的輸入有被重送。
- [ ] 輸出緩衝區大小取自 `audio_resampler_output_bound()`，不是自己按比例算的。
- [ ] `hpf` 是**單通道**模組：多通道時每個通道各有一個 `Hpf` 實例，且已先去交錯。
- [ ] 清楚知道 `wav_read_float()` 只回第一個通道；多通道來源已在上游處理。
- [ ] 不相等取樣率時，resampler 的輸入與輸出緩衝區不重疊。

**生命週期**

- [ ] 所有有狀態的物件（`Hpf`、`AudioResampler`、`FftHandle`、`AcStft`、`AcIstft`）在整段串流期間存活，不是每個區塊重建。
- [ ] `AcStft`／`AcIstft` 的 `center`、`ola`、`spec_scale`／`ispec_scale` 在建立時決定；`center = 1` 時 `ac_stft_push` 的輸出緩衝區容納 2 幀。
- [ ] `audio_resampler_reset()` 只在真的要丟棄串流歷史時呼叫。
- [ ] 每個 handle 不會被兩條執行緒同時使用。
- [ ] `wav_finalize_write()` 對每個 writer 只呼叫一次，之後不再碰該指標。
> [索引](c_user_manual_zh_TW.md) | [上一部分](c_user_manual_zh_TW_02_modules_util_io.md) <!-- split-nav -->
