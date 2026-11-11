#ifndef TFLITE_PATCH_H
#define TFLITE_PATCH_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define TFLITE_PATCH_EXPORT __attribute__((visibility("default")))
#else
#define TFLITE_PATCH_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Default offsets for TFLite internal structures */
#define DEFAULT_INTERP_SUBGRAPH_LOC 0x68
#define DEFAULT_SG_TENSOR_ARRAY     0x08
#define DEFAULT_SG_TENSOR_COUNT     0x10

#define DEFAULT_TENSOR_TYPE         0x00
#define DEFAULT_TENSOR_DATA         0x08
#define DEFAULT_TENSOR_DIMS         0x10
#define DEFAULT_TENSOR_BYTES        0x28
#define DEFAULT_TENSOR_SIZE         0x70

#define MAX_SAVED_INSN_BYTES 32

#define MAX_TRACE_PATH 512
#define MAX_MODELS 128
#define MAX_MODEL_METADATA 256
#define MAX_MODEL_NAME 512
#define MAX_BOUNDARY_TENSORS 64
#define MAX_CUSTOM_TENSORS 128
#define MAX_SHAPE_TEXT 96
#define MAX_TENSOR_NAME 256
#define MAX_DUMP_PATH 512
#define MAX_TENSOR_OVERRIDES 32
#define MAX_LATENCY_SAMPLES 10000

#define DEFAULT_TENSOR_LIMIT 128
#define DEFAULT_MAX_TENSOR_BYTES (16 * 1024 * 1024)

/* TFLite Data Types matching TfLiteType enum */
enum {
    TFLITE_TYPE_NONE = 0,
    TFLITE_TYPE_FLOAT32 = 1,
    TFLITE_TYPE_INT32 = 2,
    TFLITE_TYPE_UINT8 = 3,
    TFLITE_TYPE_INT64 = 4,
    TFLITE_TYPE_STRING = 5,
    TFLITE_TYPE_BOOL = 6,
    TFLITE_TYPE_INT16 = 7,
    TFLITE_TYPE_COMPLEX64 = 8,
    TFLITE_TYPE_INT8 = 9,
    TFLITE_TYPE_FLOAT16 = 10,
    TFLITE_TYPE_FLOAT64 = 11,
    TFLITE_TYPE_COMPLEX128 = 12,
    TFLITE_TYPE_UINT64 = 13,
    TFLITE_TYPE_RESOURCE = 14,
    TFLITE_TYPE_VARIANT = 15,
    TFLITE_TYPE_UINT32 = 16,
    TFLITE_TYPE_UINT16 = 17,
    TFLITE_TYPE_INT4 = 18,
    TFLITE_TYPE_BFLOAT16 = 19,
};

/* Fault injection modes */
enum {
    FAULT_NONE = 0,
    FAULT_BIT_FLIP = 1,
    FAULT_NOISE = 2,
    FAULT_ZERO_OUT = 3,
    FAULT_NAN_INJECT = 4,
};

/* Model snapshot descriptor filled during Invoke execution */
struct model_snapshot {
    int id;
    int subgraph_index;
    uint64_t tensor_count;
    char name[MAX_MODEL_NAME];
    uint32_t inputs[MAX_BOUNDARY_TENSORS];
    char input_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t input_count;
    uint32_t outputs[MAX_BOUNDARY_TENSORS];
    char output_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t output_count;
    int metadata_found;
    int ambiguous_model_path;
    int is_new;
    int tensor_count_changed;
};

/* Configurable memory layout offsets */
struct interpreter_layout {
    size_t interp_subgraph_loc;
    size_t sg_tensor_array;
    size_t sg_tensor_count;
    size_t tensor_type;
    size_t tensor_data;
    size_t tensor_dims;
    size_t tensor_bytes;
    size_t tensor_size;
};

/* In-memory zero-copy tensor callback signature */
typedef void (*tflite_tensor_callback_t)(
    int call_id,
    const char* stage,       /* "PRE" or "POST" */
    const char* model_name,
    int model_id,
    int subgraph_index,
    uint32_t tensor_idx,
    const char* tensor_name,
    int dtype,
    const char* shape_text,
    void* data_ptr,
    size_t byte_size,
    double elapsed_ms
);

/* Latency profiling summary */
struct latency_profile {
    uint64_t total_calls;
    double avg_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p90_ms;
    double p99_ms;
    uint64_t nan_inf_count;
};

/* Global runtime settings */
extern int g_tensor_limit;
extern size_t g_max_tensor_bytes;
extern int g_preview_values;
extern int g_tensor_name_logging;
extern int g_trace_all_tensors;
extern int g_target_subgraph;
extern int g_fault_mode;
extern float g_fault_param;
extern int g_fault_tensor_idx;
extern struct interpreter_layout g_interp_layout;

/* Internal helpers */
int plausible_ptr(const void* p);
double now_realtime(void);
double now_monotonic(void);
void tracef(const char* tag, const char* fmt, ...);

/* --- Configuration API --- */
TFLITE_PATCH_EXPORT void set_trace_output_path(const char* path);
TFLITE_PATCH_EXPORT void set_tensor_trace_limit(int limit);
TFLITE_PATCH_EXPORT void set_tensor_max_bytes(size_t max_bytes);
TFLITE_PATCH_EXPORT void set_tensor_preview_values(int count);
TFLITE_PATCH_EXPORT void set_tensor_name_logging(int enabled);
TFLITE_PATCH_EXPORT void set_trace_all_tensors(int enabled);
TFLITE_PATCH_EXPORT void set_target_subgraph(int subgraph_idx);
TFLITE_PATCH_EXPORT void set_custom_tensor_indices(const uint32_t* indices, uint32_t count);
TFLITE_PATCH_EXPORT void clear_custom_tensor_indices(void);
TFLITE_PATCH_EXPORT void set_interpreter_layout(size_t subgraph_offset,
                                                size_t tensor_array_offset,
                                                size_t tensor_count_offset,
                                                size_t tensor_size);
TFLITE_PATCH_EXPORT int get_invoke_call_count(void);

/* --- Fault Injection & Chaos Testing API --- */
TFLITE_PATCH_EXPORT void set_fault_injection(int mode, float param, int tensor_idx);
TFLITE_PATCH_EXPORT void clear_fault_injection(void);

/* --- In-Memory Callbacks API --- */
TFLITE_PATCH_EXPORT void set_tensor_callback(tflite_tensor_callback_t cb);
TFLITE_PATCH_EXPORT void clear_tensor_callback(void);

/* --- Live Tensor Override / Mutation API --- */
TFLITE_PATCH_EXPORT int set_tensor_override(const char* model_name,
                                            int is_post,
                                            uint32_t tensor_idx,
                                            const void* data,
                                            size_t byte_size);
TFLITE_PATCH_EXPORT void clear_tensor_overrides(void);

/* --- Performance Profiling & Latency Analytics API --- */
TFLITE_PATCH_EXPORT void get_latency_profile(const char* model_name, struct latency_profile* out);
TFLITE_PATCH_EXPORT void reset_latency_profile(void);

/* --- Op-Level Micro-Profiling API --- */
#define MAX_OP_PROFILES 256
struct op_profile_entry {
    char op_name[64];
    uint32_t execution_count;
    double total_time_ms;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
};

TFLITE_PATCH_EXPORT void record_op_execution(const char* op_name, double duration_ms);
TFLITE_PATCH_EXPORT int get_op_profiles(struct op_profile_entry* out_entries, int max_entries);
TFLITE_PATCH_EXPORT int get_op_profile_count(void);
TFLITE_PATCH_EXPORT void reset_op_profiles(void);

/* --- Delegate & Hardware Acceleration Tracking API --- */
struct delegate_tracking_info {
    int delegate_attached;
    char delegate_name[64];
    uint32_t delegated_nodes;
    uint32_t cpu_fallback_nodes;
    double delegation_ratio;
};

TFLITE_PATCH_EXPORT void record_delegate_attachment(const char* delegate_name,
                                                    uint32_t delegated_nodes,
                                                    uint32_t cpu_fallback_nodes);
TFLITE_PATCH_EXPORT void get_delegate_info(struct delegate_tracking_info* out);
TFLITE_PATCH_EXPORT void reset_delegate_info(void);
TFLITE_PATCH_EXPORT int install_delegate_hook_symbol(const char* lib_name, const char* symbol_name);

/* --- Memory Safety Sentinel API --- */
#define MAX_CANARIES 128
#define SENTINEL_CANARY_MAGIC 0xDEADBEEFFEEDFACEULL

struct memory_health_info {
    uint64_t rss_bytes;
    uint64_t vsize_bytes;
    uint64_t heap_bytes;
    int canary_corruptions;
    int leak_flag;
    double leak_rate_kb;
};

TFLITE_PATCH_EXPORT void sentinel_get_memory_info(struct memory_health_info* out);
TFLITE_PATCH_EXPORT int sentinel_register_canary(void* buffer, size_t size);
TFLITE_PATCH_EXPORT int sentinel_verify_canaries(void);
TFLITE_PATCH_EXPORT void sentinel_reset(void);

/* --- Model Metadata API --- */
TFLITE_PATCH_EXPORT int register_model_metadata(const char* path, uint64_t tensor_count,
                                                const uint32_t* inputs, uint32_t input_count,
                                                const uint32_t* outputs, uint32_t output_count);
TFLITE_PATCH_EXPORT int register_model_metadata_with_names(const char* path, uint64_t tensor_count,
                                                           const uint32_t* inputs,
                                                           uint32_t input_count,
                                                           const char* const* input_names,
                                                           const uint32_t* outputs,
                                                           uint32_t output_count,
                                                           const char* const* output_names);
TFLITE_PATCH_EXPORT void clear_model_metadata(void);

/* --- Tensor Dump API --- */
TFLITE_PATCH_EXPORT int enable_tensor_dump(const char* root_dir, const char* model_paths_csv);
TFLITE_PATCH_EXPORT void disable_tensor_dump(void);

/* --- Hook Installation API --- */
TFLITE_PATCH_EXPORT int install_invoke_hook_symbol(const char* lib_name, const char* symbol_name);
TFLITE_PATCH_EXPORT int install_invoke_hook_address(void* target_address);
TFLITE_PATCH_EXPORT int install_invoke_hook_offset(void* base_address, uintptr_t offset);
TFLITE_PATCH_EXPORT int install_invoke_hook_relative(const char* symbol_name,
                                                    intptr_t symbol_offset,
                                                    uintptr_t target_offset);
TFLITE_PATCH_EXPORT int remove_all_hooks(void);
TFLITE_PATCH_EXPORT int is_hook_installed(void);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_PATCH_H */
