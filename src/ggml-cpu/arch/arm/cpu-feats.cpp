#include "ggml-backend-impl.h"

#if defined(__aarch64__)

#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__QNX__)
#include <setjmp.h>
#include <signal.h>

// QNX has no getauxval()/HWCAP, and reading the ID_AA64*_EL1 registers from
// EL0 raises SIGILL (Linux emulates those reads; QNX does not). Probe each
// feature by executing one instruction that needs it and catching SIGILL.
// The assembler encodes each instruction, enabled per function with a target
// attribute, so the probes do not depend on hand-written opcodes. This runs
// once, when the backend variant is scored at load time.
static sigjmp_buf ggml_qnx_probe_jmp;

static void ggml_qnx_probe_sigill(int) {
    siglongjmp(ggml_qnx_probe_jmp, 1);
}

static bool ggml_qnx_probe(void (*insn)(void)) {
    struct sigaction sa = {};
    struct sigaction old = {};
    sa.sa_handler = ggml_qnx_probe_sigill;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGILL, &sa, &old) != 0) {
        return false;
    }
    bool ok = false;
    if (sigsetjmp(ggml_qnx_probe_jmp, 1) == 0) {
        insn();
        ok = true;
    }
    sigaction(SIGILL, &old, nullptr);
    return ok;
}

__attribute__((target("+dotprod"))) static void ggml_qnx_probe_dotprod(void) {
    __asm__ volatile("sdot v0.4s, v1.16b, v2.16b" ::: "v0");
}
__attribute__((target("+fp16"))) static void ggml_qnx_probe_fp16(void) {
    __asm__ volatile("fadd v0.8h, v1.8h, v2.8h" ::: "v0");
}
__attribute__((target("+i8mm"))) static void ggml_qnx_probe_i8mm(void) {
    __asm__ volatile("smmla v0.4s, v1.16b, v2.16b" ::: "v0");
}
__attribute__((target("+sve"))) static void ggml_qnx_probe_sve(void) {
    __asm__ volatile("orr z0.d, z1.d, z2.d" ::: "z0");
}
__attribute__((target("+sve2"))) static void ggml_qnx_probe_sve2(void) {
    __asm__ volatile("eor3 z0.d, z0.d, z1.d, z2.d" ::: "z0");
}
#endif

#if !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1 << 1)
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1 << 13)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1 << 23)
#endif

struct aarch64_features {
    // has_neon not needed, aarch64 has NEON guaranteed
    bool has_dotprod     = false;
    bool has_fp16_va     = false;
    bool has_sve         = false;
    bool has_sve2        = false;
    bool has_i8mm        = false;
    bool has_sme         = false;

    aarch64_features() {
#if defined(__linux__)
        uint32_t hwcap = getauxval(AT_HWCAP);
        uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
        has_fp16_va = !!(hwcap & HWCAP_FPHP);
        has_sve     = !!(hwcap & HWCAP_SVE);
        has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
        has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
        has_sme     = !!(hwcap2 & HWCAP2_SME);
#elif defined(__APPLE__)
        int oldp = 0;
        size_t size = sizeof(oldp);

        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, NULL, 0) == 0) {
            has_dotprod = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) == 0) {
            has_i8mm = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) == 0) {
            has_sme = static_cast<bool>(oldp);
        }

        // Apple apparently does not implement SVE yet
#elif defined(__QNX__)
        has_dotprod = ggml_qnx_probe(ggml_qnx_probe_dotprod);
        has_fp16_va = ggml_qnx_probe(ggml_qnx_probe_fp16);
        has_i8mm    = ggml_qnx_probe(ggml_qnx_probe_i8mm);
        has_sve     = ggml_qnx_probe(ggml_qnx_probe_sve);
        has_sve2    = has_sve && ggml_qnx_probe(ggml_qnx_probe_sve2);
        // SME is left off: probing it means entering streaming mode, and no
        // QNX target with SME is available to test against.
#endif
    }
};

static int ggml_backend_cpu_aarch64_score() {
    int score = 1;
    aarch64_features af;

#ifdef GGML_USE_DOTPROD
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    if (!af.has_fp16_va) { return 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__)
