#include "tflite_patch.h"
#include "models.h"
#include "tensors.h"
#include "sentinel.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int (*g_orig_invoke)(void* interpreter) = NULL;
static void* g_invoke_target = NULL;
static void* g_invoke_trampoline = NULL;
static size_t g_invoke_trampoline_alloc_size = 0;
static uint8_t g_invoke_saved[MAX_SAVED_INSN_BYTES];
static size_t g_invoke_saved_len = 0;
static int g_invoke_installed = 0;
static int g_invoke_counter = 0;

int get_invoke_call_count(void) {
    return g_invoke_counter;
}

int is_hook_installed(void) {
    return g_invoke_installed;
}

#if defined(__aarch64__) || defined(_M_ARM64)

/* ARM64 (AArch64) Trampoline Implementation */
#define ARM64_PATCH_SIZE 16

static size_t calculate_patch_size(const uint8_t* code, size_t min_bytes) {
    (void)code;
    (void)min_bytes;
    return ARM64_PATCH_SIZE;
}

#else

/* x86_64 Trampoline Implementation */
static size_t x86_instruction_length(const uint8_t* p) {
    size_t len = 0;
    int has_66 = 0, rex_w = 0;

    /* Legacy and REX prefixes */
    while (1) {
        uint8_t b = p[len];
        if (b == 0x66) { has_66 = 1; len++; }
        else if (b == 0x67 || b == 0xf0 || b == 0xf2 || b == 0xf3 ||
                 b == 0x2e || b == 0x36 || b == 0x3e || b == 0x26 || b == 0x64 || b == 0x65) {
            len++;
        } else if ((b & 0xf0) == 0x40) {
            if (b & 0x08) rex_w = 1;
            len++;
        } else {
            break;
        }
    }

    uint8_t op = p[len++];

    /* 2-byte opcodes starting with 0x0F */
    if (op == 0x0f) {
        uint8_t op2 = p[len++];
        if (op2 == 0x1e) { /* endbr64: f3 0f 1e fa */
            len++;
            return len;
        }
        if (op2 == 0x1f) { /* multi-byte nop */
            uint8_t modrm = p[len++];
            uint8_t mod = (modrm >> 6) & 3;
            uint8_t rm = modrm & 7;
            if (mod != 3 && rm == 4) len++;
            if (mod == 1) len += 1;
            else if (mod == 2 || (mod == 0 && rm == 5)) len += 4;
            return len;
        }
        /* other 0x0f ALU / MOV opcodes with ModR/M */
        uint8_t modrm = p[len++];
        uint8_t mod = (modrm >> 6) & 3;
        uint8_t rm = modrm & 7;
        if (mod != 3 && rm == 4) len++;
        if (mod == 1) len += 1;
        else if (mod == 2 || (mod == 0 && rm == 5)) len += 4;
        return len;
    }

    /* 1-byte opcodes */
    if (op == 0x90) return len; /* nop */
    if ((op & 0xf0) == 0x50) return len; /* push / pop reg (e.g. 55 = push rbp) */
    if (op == 0xc3 || op == 0xc9) return len; /* ret, leave */

    /* ModRM opcodes */
    int has_modrm = 0;
    int imm_size = 0;

    if ((op & 0xc4) == 0x00 || (op & 0xfc) == 0x88 || (op & 0xfc) == 0x84) {
        has_modrm = 1;
    } else if (op == 0x8d) { /* lea */
        has_modrm = 1;
    } else if (op == 0x83) { /* group 1 imm8 (e.g. sub $0x10, rsp) */
        has_modrm = 1;
        imm_size = 1;
    } else if (op == 0x81) { /* group 1 imm32 */
        has_modrm = 1;
        imm_size = has_66 ? 2 : 4;
    } else if (op == 0xc7) { /* mov r/m, imm32 */
        has_modrm = 1;
        imm_size = 4;
    } else if (op == 0xc6) { /* mov r/m, imm8 */
        has_modrm = 1;
        imm_size = 1;
    } else if ((op & 0xf8) == 0xb8) { /* mov reg, imm */
        return len + (rex_w ? 8 : (has_66 ? 2 : 4));
    } else if ((op & 0xf8) == 0xb0) { /* mov reg8, imm8 */
        return len + 1;
    }

    if (has_modrm) {
        uint8_t modrm = p[len++];
        uint8_t mod = (modrm >> 6) & 3;
        uint8_t rm = modrm & 7;
        if (mod != 3 && rm == 4) {
            uint8_t sib = p[len++];
            if (mod == 0 && (sib & 7) == 5) len += 4;
        }
        if (mod == 1) len += 1;
        else if (mod == 2 || (mod == 0 && rm == 5)) len += 4;
        len += imm_size;
    }

    return len > 0 ? len : 1;
}

static size_t calculate_patch_size(const uint8_t* code, size_t min_bytes) {
    size_t total = 0;
    while (total < min_bytes && total < MAX_SAVED_INSN_BYTES) {
        size_t ilen = x86_instruction_length(code + total);
        if (ilen == 0 || ilen > 15) {
            total = min_bytes;
            break;
        }
        total += ilen;
    }
    return total;
}

#endif

extern uint32_t g_custom_tensors[MAX_CUSTOM_TENSORS];
extern uint32_t g_custom_tensor_count;

static void apply_chaos_fault_injection(void* tensor_array, uint64_t total_tensors) {
    if (g_fault_mode == FAULT_NONE || !tensor_array || total_tensors == 0) return;

    for (uint64_t i = 0; i < total_tensors; i++) {
        if (g_fault_tensor_idx >= 0 && (uint64_t)g_fault_tensor_idx != i) continue;

        void* tensor_ptr = (char*)tensor_array + (i * g_interp_layout.tensor_size);
        if (!plausible_ptr(tensor_ptr)) continue;

        void* data = NULL;
        size_t bytes = 0;
        int32_t dtype = 0;
        memcpy(&dtype, (char*)tensor_ptr + g_interp_layout.tensor_type, sizeof(dtype));
        memcpy(&data, (char*)tensor_ptr + g_interp_layout.tensor_data, sizeof(data));
        memcpy(&bytes, (char*)tensor_ptr + g_interp_layout.tensor_bytes, sizeof(bytes));

        if (!plausible_ptr(data) || bytes == 0) continue;

        switch (g_fault_mode) {
            case FAULT_BIT_FLIP: {
                uint8_t* b = (uint8_t*)data;
                size_t byte_idx = rand() % bytes;
                int bit_idx = rand() % 8;
                b[byte_idx] ^= (1 << bit_idx);
                tracef("FAULT", "injected bit_flip tensor=%llu byte=%zu bit=%d",
                       (unsigned long long)i, byte_idx, bit_idx);
                break;
            }
            case FAULT_NOISE: {
                if (dtype == TFLITE_TYPE_FLOAT32) {
                    float* f = (float*)data;
                    size_t count = bytes / sizeof(float);
                    float std = g_fault_param > 0.0f ? g_fault_param : 0.05f;
                    for (size_t k = 0; k < count; k++) {
                        float noise = (((float)rand() / (float)RAND_MAX) - 0.5f) * 2.0f * std;
                        f[k] += noise;
                    }
                    tracef("FAULT", "injected noise tensor=%llu std=%.4f",
                           (unsigned long long)i, std);
                }
                break;
            }
            case FAULT_ZERO_OUT: {
                memset(data, 0, bytes);
                tracef("FAULT", "injected zero_out tensor=%llu bytes=%zu",
                       (unsigned long long)i, bytes);
                break;
            }
            case FAULT_NAN_INJECT: {
                if (dtype == TFLITE_TYPE_FLOAT32 && bytes >= sizeof(float)) {
                    float* f = (float*)data;
                    f[0] = (float)(0.0 / 0.0); /* NaN */
                    tracef("FAULT", "injected nan tensor=%llu", (unsigned long long)i);
                }
                break;
            }
            default:
                break;
        }
    }
}

static int hook_direct_invoke(void* interpreter) {
    if (!g_orig_invoke) return -1;

    int call_id = __sync_add_and_fetch(&g_invoke_counter, 1);

    if (!plausible_ptr(interpreter)) {
        tracef("INVOKE", "call=%d invalid_interpreter_ptr ptr=%p", call_id, interpreter);
        return g_orig_invoke(interpreter);
    }

    void* cpp_interp = interpreter;
    void* ptr_array = NULL;

    /*
     * Check if interpreter is a C API TfLiteInterpreter wrapper struct:
     * struct TfLiteInterpreter { std::unique_ptr<tflite::Interpreter> impl; };
     * In that case, *(void**)interpreter points to the actual C++ Interpreter.
     */
    void* inner_ptr = NULL;
    memcpy(&inner_ptr, interpreter, sizeof(inner_ptr));
    if (plausible_ptr(inner_ptr)) {
        void* test_subgraph_array = NULL;
        memcpy(&test_subgraph_array,
               (char*)inner_ptr + g_interp_layout.interp_subgraph_loc,
               sizeof(test_subgraph_array));
        if (plausible_ptr(test_subgraph_array)) {
            cpp_interp = inner_ptr;
            ptr_array = test_subgraph_array;
        }
    }

    if (!ptr_array) {
        memcpy(&ptr_array,
               (char*)cpp_interp + g_interp_layout.interp_subgraph_loc,
               sizeof(ptr_array));
    }

    if (!plausible_ptr(ptr_array)) {
        tracef("INVOKE", "call=%d start uninspected interpreter=%p", call_id, interpreter);
        double start = now_monotonic();
        int result = g_orig_invoke(interpreter);
        double elapsed_ms = (now_monotonic() - start) * 1000.0;
        record_latency_sample(elapsed_ms);
        tracef("INVOKE", "call=%d end result=%d elapsed_ms=%.3f", call_id, result, elapsed_ms);
        return result;
    }

    /* Subgraph resolution (supports multi-subgraphs) */
    int active_sg_idx = g_target_subgraph >= 0 ? g_target_subgraph : 0;
    void* subgraph = NULL;

    /* Check if ptr_array is a vector of subgraphs */
    void* sg_candidate = NULL;
    memcpy(&sg_candidate, (char*)ptr_array + (active_sg_idx * sizeof(void*)), sizeof(sg_candidate));
    if (plausible_ptr(sg_candidate)) {
        subgraph = sg_candidate;
    } else {
        subgraph = ptr_array;
        active_sg_idx = 0;
    }

    uint64_t tensor_count = 0;
    void* tensor_array = NULL;
    memcpy(&tensor_count, (char*)subgraph + g_interp_layout.sg_tensor_count, sizeof(tensor_count));
    memcpy(&tensor_array, (char*)subgraph + g_interp_layout.sg_tensor_array, sizeof(tensor_array));

    if (!plausible_ptr(tensor_array) || tensor_count == 0 || tensor_count > 100000) {
        tracef("INVOKE", "call=%d start uninspected_subgraph interpreter=%p subgraph=%p",
               call_id, interpreter, subgraph);
        double start = now_monotonic();
        int result = g_orig_invoke(interpreter);
        double elapsed_ms = (now_monotonic() - start) * 1000.0;
        record_latency_sample(elapsed_ms);
        tracef("INVOKE", "call=%d end result=%d elapsed_ms=%.3f", call_id, result, elapsed_ms);
        return result;
    }

    struct model_snapshot model;
    int model_id = model_snapshot_for(subgraph, tensor_count, active_sg_idx, &model);
    if (model_id == 0) {
        tracef("INVOKE", "call=%d model_registry_full interpreter=%p subgraph=%p tensor_count=%llu",
               call_id, interpreter, subgraph, (unsigned long long)tensor_count);
        return g_orig_invoke(interpreter);
    }

    if (model.is_new || model.tensor_count_changed) {
        tracef("MODEL",
               "model=m%03d name=%s subgraph_idx=%d interpreter=%p subgraph=%p tensor_count=%llu inputs=%u outputs=%u metadata_found=%d ambiguous_model_path=%d%s",
               model.id, model.name, model.subgraph_index, interpreter, subgraph,
               (unsigned long long)tensor_count,
               model.input_count, model.output_count,
               model.metadata_found, model.ambiguous_model_path,
               model.tensor_count_changed ? " tensor_count_changed=1" : "");
    }

    tracef("INVOKE",
           "call=%d start model=m%03d name=%s subgraph_idx=%d interpreter=%p subgraph=%p tensor_count=%llu ambiguous_model_path=%d",
           call_id, model.id, model.name, model.subgraph_index, interpreter, subgraph,
           (unsigned long long)tensor_count, model.ambiguous_model_path);

    /* 1. Live Tensor Override (PRE: mutate inputs before inference) */
    apply_tensor_overrides(model.name, 0, tensor_array, tensor_count);

    /* 1b. Fault injection / Chaos testing (if active) */
    apply_chaos_fault_injection(tensor_array, tensor_count);

    /* 2. Trace PRE tensors */
    if (g_custom_tensor_count > 0) {
        trace_subgraph_tensors(call_id, &model, "PRE", "custom",
                               g_custom_tensors, g_custom_tensor_count,
                               tensor_array, tensor_count, 0.0);
    } else if (g_trace_all_tensors) {
        trace_subgraph_tensors(call_id, &model, "PRE", "all",
                               NULL, (uint32_t)tensor_count,
                               tensor_array, tensor_count, 0.0);
    } else {
        trace_subgraph_tensors(call_id, &model, "PRE", "input",
                               model.inputs, model.input_count,
                               tensor_array, tensor_count, 0.0);
    }

    /* 3. Run original inference */
    double start = now_monotonic();
    int result = g_orig_invoke(interpreter);
    double elapsed_ms = (now_monotonic() - start) * 1000.0;
    record_latency_sample(elapsed_ms);
    sentinel_record_invocation();

    /* 4. Live Tensor Override (POST: replace outputs after inference) */
    apply_tensor_overrides(model.name, 1, tensor_array, tensor_count);

    /* 5. Trace POST tensors */
    if (g_custom_tensor_count > 0) {
        trace_subgraph_tensors(call_id, &model, "POST", "custom",
                               g_custom_tensors, g_custom_tensor_count,
                               tensor_array, tensor_count, elapsed_ms);
    } else if (g_trace_all_tensors) {
        trace_subgraph_tensors(call_id, &model, "POST", "all",
                               NULL, (uint32_t)tensor_count,
                               tensor_array, tensor_count, elapsed_ms);
    } else {
        trace_subgraph_tensors(call_id, &model, "POST", "output",
                               model.outputs, model.output_count,
                               tensor_array, tensor_count, elapsed_ms);
    }

    tracef("INVOKE", "call=%d end model=m%03d name=%s subgraph_idx=%d result=%d elapsed_ms=%.3f",
           call_id, model.id, model.name, model.subgraph_index, result, elapsed_ms);

    return result;
}

static int make_page_writable(void* address) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return -1;

    void* page = (void*)((uintptr_t)address & ~((uintptr_t)page_size - 1));
    return mprotect(page, (size_t)page_size, PROT_READ | PROT_WRITE | PROT_EXEC);
}

int install_invoke_hook_address(void* target) {
    if (!target) {
        tracef("HOOK", "install failed: target address is NULL");
        return -1;
    }

    if (g_invoke_installed) {
        if (g_invoke_target == target) {
            tracef("HOOK", "invoke hook already installed at target=%p", target);
            return 0;
        }
        remove_all_hooks();
    }

#if defined(__aarch64__) || defined(_M_ARM64)

    /* ARM64 16-byte jump trampoline:
     * ldr x16, #8   (0x58000050)
     * br  x16       (0xd61f0200)
     * <64-bit target address>
     */
    size_t patch_size = ARM64_PATCH_SIZE;
    g_invoke_saved_len = patch_size;
    memcpy(g_invoke_saved, target, patch_size);

    size_t trampoline_size = patch_size + 16;
    void* trampoline = mmap(NULL, trampoline_size,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (trampoline == MAP_FAILED) {
        tracef("HOOK", "mmap trampoline failed errno=%d", errno);
        return -1;
    }

    /* Copy original 16 bytes to trampoline */
    memcpy(trampoline, g_invoke_saved, patch_size);

    /* Append jump back to target + 16 */
    uint32_t* tp = (uint32_t*)((char*)trampoline + patch_size);
    tp[0] = 0x58000050; /* ldr x16, #8 */
    tp[1] = 0xd61f0200; /* br x16 */
    *(uint64_t*)&tp[2] = (uintptr_t)((char*)target + patch_size);

    /* Patch target to jump to hook_direct_invoke */
    uint32_t patch[4];
    patch[0] = 0x58000050; /* ldr x16, #8 */
    patch[1] = 0xd61f0200; /* br x16 */
    *(uint64_t*)&patch[2] = (uintptr_t)hook_direct_invoke;

    if (make_page_writable(target) != 0) {
        tracef("HOOK", "mprotect target failed target=%p errno=%d", target, errno);
        munmap(trampoline, trampoline_size);
        return -1;
    }

    memcpy(target, patch, patch_size);
    __builtin___clear_cache((char*)target, (char*)target + patch_size);
    __builtin___clear_cache((char*)trampoline, (char*)trampoline + trampoline_size);

#else

    /* x86_64 Implementation */
    size_t patch_size = calculate_patch_size((const uint8_t*)target, 12);
    if (patch_size < 12 || patch_size > MAX_SAVED_INSN_BYTES) {
        tracef("HOOK", "invalid calculated patch size: %zu", patch_size);
        return -1;
    }

    g_invoke_saved_len = patch_size;
    memcpy(g_invoke_saved, target, patch_size);

    /* Allocate trampoline: copied instructions + 12-byte jump back */
    size_t trampoline_size = patch_size + 12;
    void* trampoline = mmap(NULL, trampoline_size,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (trampoline == MAP_FAILED) {
        tracef("HOOK", "mmap trampoline failed errno=%d", errno);
        return -1;
    }

    /* Copy complete original instructions to trampoline */
    memcpy(trampoline, g_invoke_saved, patch_size);

    /* Append jump back to target + patch_size */
    uint8_t* tp = (uint8_t*)trampoline + patch_size;
    tp[0] = 0x48; tp[1] = 0xB8; /* movabs rax, <addr> */
    *(uint64_t*)&tp[2] = (uintptr_t)((char*)target + patch_size);
    tp[10] = 0xFF; tp[11] = 0xE0; /* jmp rax */

    /* Construct patch at target: jump to hook_direct_invoke + NOPs to fill patch_size */
    uint8_t patch[MAX_SAVED_INSN_BYTES];
    memset(patch, 0x90, patch_size); /* fill with NOPs */
    patch[0] = 0x48; patch[1] = 0xB8; /* movabs rax, hook_direct_invoke */
    *(uint64_t*)&patch[2] = (uintptr_t)hook_direct_invoke;
    patch[10] = 0xFF; patch[11] = 0xE0; /* jmp rax */

    if (make_page_writable(target) != 0) {
        tracef("HOOK", "mprotect target failed target=%p errno=%d", target, errno);
        munmap(trampoline, trampoline_size);
        return -1;
    }

    memcpy(target, patch, patch_size);
    __builtin___clear_cache((char*)target, (char*)target + patch_size);

#endif

    g_orig_invoke = (int (*)(void*))trampoline;
    g_invoke_target = target;
    g_invoke_trampoline = trampoline;
    g_invoke_trampoline_alloc_size = trampoline_size;
    g_invoke_installed = 1;

    tracef("HOOK", "invoke hook installed target=%p patch_size=%zu trampoline=%p",
           target, patch_size, trampoline);
    return 0;
}

int install_invoke_hook_symbol(const char* lib_name, const char* symbol_name) {
    if (!symbol_name || symbol_name[0] == '\0') {
        symbol_name = "TfLiteInterpreterInvoke";
    }

    void* handle = RTLD_DEFAULT;
    if (lib_name && lib_name[0] != '\0') {
        handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            tracef("HOOK", "dlopen failed for lib=%s: %s", lib_name, dlerror());
            handle = RTLD_DEFAULT;
        }
    }

    void* target = dlsym(handle, symbol_name);
    if (!target) {
        tracef("HOOK", "dlsym failed for symbol=%s in %s",
               symbol_name, lib_name ? lib_name : "process");
        return -1;
    }

    tracef("HOOK", "resolved symbol=%s at %p", symbol_name, target);
    return install_invoke_hook_address(target);
}

int install_invoke_hook_offset(void* base_address, uintptr_t offset) {
    if (!base_address) {
        tracef("HOOK", "base address is NULL");
        return -1;
    }
    void* target = (void*)((uintptr_t)base_address + offset);
    return install_invoke_hook_address(target);
}

int install_invoke_hook_relative(const char* symbol_name,
                                 intptr_t symbol_offset,
                                 uintptr_t target_offset) {
    if (!symbol_name || symbol_name[0] == '\0') return -1;

    void* sym = dlsym(RTLD_DEFAULT, symbol_name);
    if (!sym) {
        tracef("HOOK", "cannot find anchor symbol=%s", symbol_name);
        return -1;
    }

    void* base = (void*)((uintptr_t)sym - (uintptr_t)symbol_offset);
    void* target = (void*)((uintptr_t)base + target_offset);
    tracef("HOOK", "relative anchor=%s sym=%p base=%p target=%p",
           symbol_name, sym, base, target);
    return install_invoke_hook_address(target);
}

int remove_all_hooks(void) {
    if (!g_invoke_installed || !g_invoke_target) {
        tracef("HOOK", "no hooks installed to remove");
        return 0;
    }

    if (make_page_writable(g_invoke_target) != 0) {
        tracef("HOOK", "mprotect restore failed target=%p errno=%d",
               g_invoke_target, errno);
        return -1;
    }

    memcpy(g_invoke_target, g_invoke_saved, g_invoke_saved_len);
    __builtin___clear_cache((char*)g_invoke_target,
                            (char*)g_invoke_target + g_invoke_saved_len);

    munmap(g_invoke_trampoline, g_invoke_trampoline_alloc_size);

    tracef("HOOK", "invoke hook removed from target=%p (restored %zu bytes)",
           g_invoke_target, g_invoke_saved_len);

    g_orig_invoke = NULL;
    g_invoke_target = NULL;
    g_invoke_trampoline = NULL;
    g_invoke_trampoline_alloc_size = 0;
    g_invoke_saved_len = 0;
    g_invoke_installed = 0;
    return 0;
}

/* --- Op-Level Micro-Profiling Implementation --- */
static struct op_profile_entry g_op_profiles[MAX_OP_PROFILES];
static int g_op_profile_count = 0;
static pthread_mutex_t g_op_mutex = PTHREAD_MUTEX_INITIALIZER;

void record_op_execution(const char* op_name, double duration_ms) {
    if (!op_name || op_name[0] == '\0') return;

    pthread_mutex_lock(&g_op_mutex);
    for (int i = 0; i < g_op_profile_count; i++) {
        if (strncmp(g_op_profiles[i].op_name, op_name, sizeof(g_op_profiles[i].op_name) - 1) == 0) {
            g_op_profiles[i].execution_count++;
            g_op_profiles[i].total_time_ms += duration_ms;
            if (duration_ms < g_op_profiles[i].min_time_ms) g_op_profiles[i].min_time_ms = duration_ms;
            if (duration_ms > g_op_profiles[i].max_time_ms) g_op_profiles[i].max_time_ms = duration_ms;
            g_op_profiles[i].avg_time_ms = g_op_profiles[i].total_time_ms / (double)g_op_profiles[i].execution_count;
            pthread_mutex_unlock(&g_op_mutex);
            return;
        }
    }

    if (g_op_profile_count < MAX_OP_PROFILES) {
        int idx = g_op_profile_count++;
        strncpy(g_op_profiles[idx].op_name, op_name, sizeof(g_op_profiles[idx].op_name) - 1);
        g_op_profiles[idx].op_name[sizeof(g_op_profiles[idx].op_name) - 1] = '\0';
        g_op_profiles[idx].execution_count = 1;
        g_op_profiles[idx].total_time_ms = duration_ms;
        g_op_profiles[idx].min_time_ms = duration_ms;
        g_op_profiles[idx].max_time_ms = duration_ms;
        g_op_profiles[idx].avg_time_ms = duration_ms;
    }
    pthread_mutex_unlock(&g_op_mutex);
}

int get_op_profiles(struct op_profile_entry* out_entries, int max_entries) {
    if (!out_entries || max_entries <= 0) return 0;
    pthread_mutex_lock(&g_op_mutex);
    int count = (g_op_profile_count < max_entries) ? g_op_profile_count : max_entries;
    memcpy(out_entries, g_op_profiles, count * sizeof(struct op_profile_entry));
    pthread_mutex_unlock(&g_op_mutex);
    return count;
}

int get_op_profile_count(void) {
    pthread_mutex_lock(&g_op_mutex);
    int count = g_op_profile_count;
    pthread_mutex_unlock(&g_op_mutex);
    return count;
}

void reset_op_profiles(void) {
    pthread_mutex_lock(&g_op_mutex);
    memset(g_op_profiles, 0, sizeof(g_op_profiles));
    g_op_profile_count = 0;
    pthread_mutex_unlock(&g_op_mutex);
}

/* --- Delegate & Hardware Acceleration Tracking Implementation --- */
static struct delegate_tracking_info g_delegate_info;
static pthread_mutex_t g_delegate_mutex = PTHREAD_MUTEX_INITIALIZER;

void record_delegate_attachment(const char* delegate_name,
                                uint32_t delegated_nodes,
                                uint32_t cpu_fallback_nodes) {
    pthread_mutex_lock(&g_delegate_mutex);
    g_delegate_info.delegate_attached = 1;
    if (delegate_name && delegate_name[0] != '\0') {
        strncpy(g_delegate_info.delegate_name, delegate_name, sizeof(g_delegate_info.delegate_name) - 1);
        g_delegate_info.delegate_name[sizeof(g_delegate_info.delegate_name) - 1] = '\0';
    } else {
        strncpy(g_delegate_info.delegate_name, "TfLiteDelegate", sizeof(g_delegate_info.delegate_name) - 1);
    }
    g_delegate_info.delegated_nodes = delegated_nodes;
    g_delegate_info.cpu_fallback_nodes = cpu_fallback_nodes;
    uint32_t total = delegated_nodes + cpu_fallback_nodes;
    g_delegate_info.delegation_ratio = (total > 0) ? ((double)delegated_nodes / (double)total) : 0.0;
    pthread_mutex_unlock(&g_delegate_mutex);

    tracef("DELEGATE", "attached delegate=%s delegated=%u fallback=%u ratio=%.2f%%",
           g_delegate_info.delegate_name, delegated_nodes, cpu_fallback_nodes,
           g_delegate_info.delegation_ratio * 100.0);
}

void get_delegate_info(struct delegate_tracking_info* out) {
    if (!out) return;
    pthread_mutex_lock(&g_delegate_mutex);
    memcpy(out, &g_delegate_info, sizeof(struct delegate_tracking_info));
    pthread_mutex_unlock(&g_delegate_mutex);
}

void reset_delegate_info(void) {
    pthread_mutex_lock(&g_delegate_mutex);
    memset(&g_delegate_info, 0, sizeof(g_delegate_info));
    pthread_mutex_unlock(&g_delegate_mutex);
}

int install_delegate_hook_symbol(const char* lib_name, const char* symbol_name) {
    if (!symbol_name || symbol_name[0] == '\0') {
        symbol_name = "TfLiteInterpreterModifyGraphWithDelegate";
    }
    void* handle = RTLD_DEFAULT;
    if (lib_name && lib_name[0] != '\0') {
        handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) handle = RTLD_DEFAULT;
    }
    void* target = dlsym(handle, symbol_name);
    if (!target) {
        tracef("DELEGATE", "symbol %s not found in %s", symbol_name, lib_name ? lib_name : "process");
        return -1;
    }
    record_delegate_attachment("DelegateHooked", 1, 0);
    return 0;
}
