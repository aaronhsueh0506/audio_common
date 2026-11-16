# libaudio_common

Build this archive with `make lib` in this directory using an ARM32 or ARM64 toolchain. Run `make -s print-lib-path` with the same options when linking your application. A workspace path with spaces builds; a quote in it makes `make -s print-lib-path` fail. The flag variables (`EXTRA_CFLAGS`, `EXTRA_LDFLAGS`) take only letters, digits and `_ . / = , + -`, separated by spaces, so a quoted token or an `-I` path with a space stops the Makefile with an FP policy conflict.

- [Offline DSP guide and streaming examples](../docs/html/dsp/index.html)
- [C integration and per-function reference](docs/c_user_manual_zh_TW.md)
