# 通用 DSP 的 C 整合

FFT、STFT、重取樣和 HPF 的輸入輸出是 float32。每條串流建立自己的處理物件，跨 hop 保留狀態。分幀與重疊相加的完整範例、啟動延遲與收尾順序見 [DSP 離線指南](../../docs/html/dsp/index.html)。

## 建置和連結

從 audiolib 根目錄執行下列命令，將 C 範例存成 `example.c`。目標編譯器須支援 ARM32 或 ARM64。

```sh
make -C common lib
cc -std=c99 -Icommon/include -Icommon -ffp-contract=off -fno-math-errno \
   example.c "$(make -s -C common print-lib-path)" -lm -o example
./example
```

## 建立高通濾波器

這個程式以 16 kHz 處理兩個連續的 160-sample 單聲道區塊。HPF 在第一個 hop 前建立，最後一個 hop 後釋放。

```c
#include "hpf.h"

int main(void)
{
    Hpf *filter = hpf_create(80.0f, 16000);
    if (!filter) return 1;

    float audio[2][160] = {{1.0f}, {0.0f}};
    for (int hop = 0; hop < 2; ++hop)
        hpf_process(filter, audio[hop], 160);

    hpf_destroy(filter);
    return 0;
}
```

## 依任務查閱

| 任務 | 參考 |
|---|---|
| 建立 FFT、重取樣器、HPF 或自備 16-byte 對齊的 pool | [函式、緩衝區及錯誤回傳](reference/core.md) |
| 在時間域與頻域之間組幀、操作 WAV 或寫 per-bin 核心 | [STFT、檔案格式與向量函式](reference/utilities.md) |
| 改用 scalar、指定目標工具鏈或比對兩種組態的輸出 | [工具鏈與數值契約](reference/build.md) |
