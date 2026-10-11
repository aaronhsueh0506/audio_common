/* Audio_ALG/AIAEC/Align_ULCNet/ulcnet_process.h: UlcnetAnalysis, UlcnetSynthesis;
 * ulcnet_process.c: ulcnet_make_window, ulcnet_rfft, ulcnet_analysis_*, ulcnet_synthesis_* */
typedef struct UlcnetAnalysis {
    const float *window;           /* caller-owned shared sqrt-Hann table */
    FftHandle   *fft;              /* caller-owned ULCNET_N_FFT-point handle; may be
                                    * shared with the other analysis and the
                                    * synthesis (strictly sequential use
                                    * within a hop -- never concurrent)   */
    float history[ULCNET_N_FFT];   /* 最近 N_FFT 個 raw 樣本 (rolling) */
    long  hops_seen;               /* saturates at 3: 1 = first hop (no frame
                                    * yet), 2 = reflect-prefix double
                                    * emission, >= 3 = steady state */
    /* Per-call FFT scratch kept in the struct, off the stack; contents are
     * undefined between calls. */
    float   seg[ULCNET_N_FFT];     /* windowed segment; clobbered by FFT  */
    Complex spec[ULCNET_BINS];     /* RFFT output staging                 */
} UlcnetAnalysis;

typedef struct UlcnetSynthesis {
    const float *window;           /* caller-owned shared sqrt-Hann table */
    FftHandle   *fft;              /* caller-owned ULCNET_N_FFT-point handle; same
                                    * sharing contract as UlcnetAnalysis  */
    float acc[ULCNET_N_FFT];       /* overlap-add 累加器 (局部原點 = 下一段輸出) */
    float env[ULCNET_N_FFT];       /* 窗平方包絡累加器 (torch.istft 語意) */
    long  frames_seen;             /* saturates at 2: 0 = nothing accumulated,
                                    * 1 = frame 0 (inside the trimmed half
                                    * window), >= 2 = steady state */
    /* Per-call FFT scratch kept in the struct, as in UlcnetAnalysis. */
    Complex spec[ULCNET_BINS];     /* IRFFT input staging; clobbered by FFT */
    float   time[ULCNET_N_FFT];    /* IRFFT time-domain output              */
} UlcnetSynthesis;

void ulcnet_make_window(float window[ULCNET_N_FFT]) {
    int i;
    for (i = 0; i < ULCNET_N_FFT; ++i) {
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)ULCNET_N_FFT));
    }
}

/* ---- forward rFFT of one windowed ULCNET_N_FFT-sample segment ----
 * Stages the windowed copy in st->seg (struct-owned scratch; clobber
 * permitted, so fft_forward_scratch may skip any backend defensive copy).
 * `segment` may alias st->seg itself: the windowing loop is a same-index
 * read-then-write. Output ordering contract: out_re/out_im[k] = Re/Im of
 * rfft bin k, k = 0..ULCNET_BINS-1 (DC..Nyquist) -- exactly the ordering
 * the Python parity gate compares bin by bin. */
static void ulcnet_rfft(UlcnetAnalysis *st, const float *segment,
                        float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    skn_mul_f32(st->seg, segment, st->window, ULCNET_N_FFT);
    fft_forward_scratch(st->fft, st->seg, st->spec);
    skn_deinterleave_cf32(st->spec, out_re, out_im, ULCNET_BINS);
}

/* ============================== analysis ============================== */

int ulcnet_analysis_init(UlcnetAnalysis *st, FftHandle *fft,
                         const float *window) {
    if (!st || !fft || !window) return -1;
    /* The handle must match the compiled grid. */
    if (fft_get_n_freqs(fft) != ULCNET_BINS) return -1;
    memset(st, 0, sizeof(*st));
    st->fft = fft;
    st->window = window;
    return 0;
}

/* Slide the last N_FFT samples by one hop and append the new hop. Shared by
 * the centered push and the rolling push: they share the history, not the
 * schedule. */
static void history_append(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP]) {
    memmove(st->history, st->history + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memcpy(st->history + ULCNET_N_FFT - ULCNET_HOP, hop_in,
           (size_t)ULCNET_HOP * sizeof(float));
}

int ulcnet_analysis_push(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP],
                         float out_re[2][ULCNET_BINS],
                         float out_im[2][ULCNET_BINS]) {
    history_append(st, hop_in);
    /* Saturates at 3: only 1, 2 and >= 3 (and, for flush, < 2 and >= 2) are
     * distinguished, and the clamp avoids signed overflow. */
    if (st->hops_seen < 3) st->hops_seen++;

    if (st->hops_seen == 1) return 0;   /* frame 0 needs sample index HOP */

    if (st->hops_seen == 2) {
        /* history == x[0..N_FFT-1]. Frame 0 covers the reflect prefix
         * (x[HOP..1], i.e. x[1..HOP] reversed) followed by x[0..HOP-1].
         * Built directly in st->seg (allowed to alias ulcnet_rfft's
         * windowing input -- see that function's comment). */
        int i;
        for (i = 0; i < ULCNET_HOP; ++i)
            st->seg[i] = st->history[ULCNET_HOP - i];  /* x[HOP-i] */
        memcpy(st->seg + ULCNET_HOP, st->history,
               (size_t)ULCNET_HOP * sizeof(float));    /* x[0..HOP-1] */
        ulcnet_rfft(st, st->seg, out_re[0], out_im[0]);
        ulcnet_rfft(st, st->history, out_re[1], out_im[1]);
        return 2;
    }

    /* Steady state: frame k = window over the last N_FFT raw samples,
     * centred on hop grid point k*HOP. */
    ulcnet_rfft(st, st->history, out_re[0], out_im[0]);
    return 1;
}

int ulcnet_analysis_push_frame(UlcnetAnalysis *st,
                               const float hop_in[ULCNET_HOP],
                               float out_re[ULCNET_BINS],
                               float out_im[ULCNET_BINS]) {
    history_append(st, hop_in);
    /* The same windowed transform of the same history the centered push
     * ends every hop with -- hence bit-identical to its last frame. */
    ulcnet_rfft(st, st->history, out_re, out_im);
    return 1;
}

int ulcnet_analysis_flush(UlcnetAnalysis *st,
                          float out_re[2][ULCNET_BINS],
                          float out_im[2][ULCNET_BINS]) {
    int i;
    if (st->hops_seen < 2) return 0;  /* centered contract needs > half a window */
    /* Trailing reflect pad mirrors the HOP samples before the last one:
     * suffix = x[L-2], x[L-3], ..., x[L-HOP-1]. One extra frame results
     * (total frames = L/HOP + 1 for hop-aligned L): it covers
     * x[L-HOP .. L-1] followed by the first HOP suffix samples. Built in
     * st->seg (aliasing permitted -- see ulcnet_rfft). */
    memcpy(st->seg, st->history + ULCNET_N_FFT - ULCNET_HOP,
           (size_t)ULCNET_HOP * sizeof(float));        /* x[L-HOP..L-1] */
    for (i = 0; i < ULCNET_HOP; ++i)
        st->seg[ULCNET_HOP + i] = st->history[ULCNET_N_FFT - 2 - i]; /* x[L-2-i] */
    ulcnet_rfft(st, st->seg, out_re[0], out_im[0]);
    return 1;
}

/* ============================== synthesis ============================= */

int ulcnet_synthesis_init(UlcnetSynthesis *st, FftHandle *fft,
                          const float *window) {
    if (!st || !fft || !window) return -1;
    if (fft_get_n_freqs(fft) != ULCNET_BINS) return -1;
    memset(st, 0, sizeof(*st));
    st->fft = fft;
    st->window = window;
    return 0;
}

int ulcnet_synthesis_push(UlcnetSynthesis *st,
                          const float re[ULCNET_BINS],
                          const float im[ULCNET_BINS],
                          float out[ULCNET_HOP]) {
    int emitted;
    skn_interleave_cf32(re, im, st->spec, ULCNET_BINS);
    /* IRFFT: the Hermitian upper half is implied by the real transform --
     * no explicit mirror, no full complex FFT. Both wrapper backends
     * normalise by 1/N (fft_wrapper.h contract), matching torch.istft's
     * irfft. spec is struct-owned scratch, so input clobber is fine. */
    fft_inverse_scratch(st->fft, st->spec, st->time);

    /* Every frame lands at local offset 0: after each push the leading HOP
     * samples are finalized and shifted out, so the accumulator origin
     * always advances exactly one hop per frame. */
    sk_wola_accumulate_f32(st->acc, st->time, st->window, ULCNET_N_FFT);
    sk_wola_accumulate_f32(st->env, st->window, st->window, ULCNET_N_FFT);
    /* Saturates at 2: only 0, 1 and >= 2 are distinguished, and the clamp
     * avoids signed overflow. */
    if (st->frames_seen < 2) st->frames_seen++;

    emitted = 0;
    if (st->frames_seen > 1) {
        /* frame 0's finalized block lies inside the trimmed half window */
        skn_div_floor_f32(out, st->acc, st->env, ULCNET_HOP, 1e-11f);
        emitted = ULCNET_HOP;
    }
    memmove(st->acc, st->acc + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memset(st->acc + ULCNET_N_FFT - ULCNET_HOP, 0,
           (size_t)ULCNET_HOP * sizeof(float));
    memmove(st->env, st->env + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memset(st->env + ULCNET_N_FFT - ULCNET_HOP, 0,
           (size_t)ULCNET_HOP * sizeof(float));
    return emitted;
}

int ulcnet_synthesis_flush(UlcnetSynthesis *st, float out[ULCNET_N_FFT]) {
    int n;
    if (st->frames_seen == 0) return 0;
    n = ULCNET_N_FFT - ULCNET_HOP;
    skn_div_floor_f32(out, st->acc, st->env, (size_t)n, 1e-11f);
    return n;
}
