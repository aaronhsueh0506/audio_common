# 工具鏈與數值契約

[C 整合手冊](../c_user_manual_zh_TW.md)

## ARM 工具鏈

建置、查詢 archive 路徑及連結應使用同一組目標設定。例如在 common 目錄以支援 NEON/VFPv4 的 ARM32 工具鏈建置：

```sh
make CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ \
     AR=arm-linux-gnueabihf-ar RANLIB=arm-linux-gnueabihf-ranlib \
     EXTRA_CFLAGS=-mfpu=neon-vfpv4 lib print-lib-path
```

FFT 後端為 NE10。`SIMD=1` 時有 NEON 的目標可使用 NEON FFT，`SIMD=0` 使用純量 C。共享的 DSP／空間音訊向量函式在 AArch64 以及具 NEON 和融合乘加的 ARM32 上啟用 NEON。AArch32 ARMv8 可用 `-mfpu=neon-fp-armv8`。只支援一般 `-mfpu=neon` 的 ARM32 會使用共享核心的純量路徑。

archive 的路徑由編譯設定決定。交叉編譯或切換 ABI 後，重新查詢每個物件的 pool 大小，並依目前 ABI 取得的結果分配。

## 呼叫端浮點旗標

- 呼叫端需有 `-I<common>/include -I<common>` 與 `-lm`。編譯含純標頭核心的 translation unit 時，使用 `-ffp-contract=off -fno-math-errno`。
- archive 使用 `SIMD=0` 時，呼叫端的純標頭核心也使用 `-DSIMD_KERNELS_FORCE_SCALAR`。
- 使用 `EXTRA_CFLAGS` 補充 compile flags。Makefile 只接受空白分隔的簡單 flag，並檢查 fast-math 和 FP contraction 衝突。
- 透過 `make -C common publish` 可取得穩定的 `common/dist/ne10/current/` 交付路徑。

使用四庫並排的 Audio_ALG 專案時，該專案的 `lib/audiolib.pin` 記錄預期版本。可在 Audio_ALG 根目錄執行 `make -C pipelines check-pins` 核對正在使用的 checkout。

## 比對輸出

同一個 ABI、FP 及 SIMD 設定下，heap 與 static pool 產生相同結果。FFT、STFT 和重取樣在 NEON 與純量設定間可有 float 捨入差異。ARMv7 NEON 的 subnormal flush 與 NaN payload 規則也會影響比較，應固定 ABI 及浮點模式後再做位元比對。

`make -C common selftest` 驗證通用 DSP。從 audiolib 根目錄執行 `make test` 可驗證空間模組（含 `audit-no-stdio`：四個封存檔不得引用 stdio 符號）。驗證共用函式庫的建置隔離與發布時，使用 `make -C common test-full`，其隔離測項需要旁邊有 AEC checkout；`test-full` 另含 `audit-fp-contract`（反組譯檢查 `-ffp-contract=off`，需要旁邊有 NR checkout）。
