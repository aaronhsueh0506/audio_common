/* Reference copies of the pipelines' synthesis hop, line-for-line. Harness stubs precede them. */
/* Audio_ALG/pipelines/mono_aec_nr_res/audio_pipeline.c:945-956 synthesize_hop, :518-519 window loop */
static void synthesize_hop(AudioPipeline* p, const Complex* spectrum,
                           float* out) {
    const int hop      = p->hop;
    const int frame_sz = p->frame_sz;
    uint32_t t2 = audio_pipeline_now_us();
    fft_inverse(p->fft, spectrum, p->ifft_buf);
    sk_wola_accumulate_f32(p->ola, p->ifft_buf, p->synth_win, frame_sz);
    memcpy(out, p->ola, (size_t)hop * sizeof(float));
    memmove(p->ola, p->ola + hop, (size_t)(frame_sz - hop) * sizeof(float));
    memset(p->ola + (frame_sz - hop), 0, (size_t)hop * sizeof(float));
    p->last_timing.synth_us = audio_pipeline_now_us() - t2;
}

static void ref_mono_window(float* w, int frame_sz) {
    struct { float* synth_win; } pp, *p = &pp;
    pp.synth_win = w;
        for (int k = 0; k < frame_sz; k++)
            p->synth_win[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * M_PI_F * k / frame_sz)));
}

/* Audio_ALG/pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c:1955-1970 finish_hop, :948-952 window loop */
static int finish_hop(FourAecNrRes* p, const Complex* spectrum, float* out) {
    int hop = p->hop_size;
    int fft = p->fft_size;
    uint32_t t0 = four_aec_nr_res_now_us();
    fft_inverse(p->fft, spectrum, p->ifft_buffer);
    sk_wola_accumulate_f32(p->ola, p->ifft_buffer, p->synth_window, fft);
    memcpy(out, p->ola, (size_t)hop * sizeof(float));
    memmove(
        p->ola, p->ola + hop,
        (size_t)(fft - hop) * sizeof(float));
    memset(
        p->ola + (fft - hop), 0,
        (size_t)hop * sizeof(float));
    p->last_timing.synth_us = four_aec_nr_res_now_us() - t0;
    return inputs_finite(out, (size_t)hop);
}

static void ref_4ch_window(float* w, int fft) {
    struct { float* synth_window; } pp, *p = &pp;
    int k;
    pp.synth_window = w;
        for (k = 0; k < fft; ++k) {
            p->synth_window[k] = sqrtf(
                0.5f * (1.0f - cosf(
                    2.0f * M_PI_F * (float)k / (float)fft)));
        }
}

/* Audio_ALG/pipelines/dfn_rate_bridge.c:385-393 synth_hop, :207-212 make_root_hann */
static void synth_hop(FftHandle *fft, const Complex *spec, float *ifft_buf,
                      float *ola, const float *win, int n_fft, int hop,
                      float *hop_out) {
    fft_inverse(fft, spec, ifft_buf);
    sk_wola_accumulate_f32(ola, ifft_buf, win, n_fft);
    memcpy(hop_out, ola, (size_t)hop * sizeof(float));
    memmove(ola, ola + hop, (size_t)(n_fft - hop) * sizeof(float));
    memset(ola + (n_fft - hop), 0, (size_t)hop * sizeof(float));
}

static void make_root_hann(float *window, int n) {
    int i;
    for (i = 0; i < n; ++i)
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)n));
}

/* Audio_ALG/lib/aec/c_impl/src/pbfdkf.c:100-103 window loop */
static void ref_pbfdkf_window(float* sqrt_hann, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        float h = 0.5f * (1.0f - cosf(2.0f * M_PI_AEC * (float)i / (float)block_size));
        sqrt_hann[i] = sqrtf(h);
    }
}
