# audiolib

DSP libraries for ARM32 and ARM64. The common library provides FFT, STFT/ISTFT, resampling and high-pass filtering. The spatial libraries run on spectra supplied by the application. VAD reads one channel, while SRP-PHAT and the beamformer read the microphone array.

From the repository root, build the libraries your application uses. A workspace path with spaces builds; a quote in it makes `make -s print-lib-path` fail. The flag variables (`EXTRA_CFLAGS`, `EXTRA_LDFLAGS`) take only letters, digits and `_ . / = , + -`, separated by spaces, so a quoted token or an `-I` path with a space stops the Makefile with an FP policy conflict.

```sh
make -C common lib
make -s -C common print-lib-path
make libspatial_common.a libdoa.a libbf.a libvad.a
make -s print-bin-dir
```

Run `make test` for the spatial libraries and `make -C common selftest` for the common DSP layer.

The offline guide covers per-hop C calls, FFT and overlap-add latency, caller-owned memory, and spatial configuration. [audiolib documentation](docs/html/index.html)
