#include "tflite_patch.h"

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t g_trace_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_trace_path[MAX_TRACE_PATH] = {0};

int g_tensor_limit = DEFAULT_TENSOR_LIMIT;
size_t g_max_tensor_bytes = DEFAULT_MAX_TENSOR_BYTES;
int g_preview_values = 0;
int g_tensor_name_logging = 1;
int g_trace_all_tensors = 0;
int g_target_subgraph = -1; /* -1 means all or default */

/* Fault injection configuration */
int g_fault_mode = FAULT_NONE;
float g_fault_param = 0.0f;
int g_fault_tensor_idx = -1; /* -1 = all inputs */

uint32_t g_custom_tensors[MAX_CUSTOM_TENSORS];
uint32_t g_custom_tensor_count = 0;

/* Latency profiling storage */
static double g_latency_history[MAX_LATENCY_SAMPLES];
static size_t g_latency_count = 0;
static uint64_t g_nan_inf_counter = 0;

struct interpreter_layout g_interp_layout = {
    .interp_subgraph_loc = DEFAULT_INTERP_SUBGRAPH_LOC,
    .sg_tensor_array = DEFAULT_SG_TENSOR_ARRAY,
    .sg_tensor_count = DEFAULT_SG_TENSOR_COUNT,
    .tensor_type = DEFAULT_TENSOR_TYPE,
    .tensor_data = DEFAULT_TENSOR_DATA,
    .tensor_dims = DEFAULT_TENSOR_DIMS,
    .tensor_bytes = DEFAULT_TENSOR_BYTES,
    .tensor_size = DEFAULT_TENSOR_SIZE,
};

int plausible_ptr(const void* p) {
    return p && (uintptr_t)p > 0x10000ULL;
}

double now_realtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1000000000.0);
}

double now_monotonic(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1000000000.0);
}

static void ensure_trace_path_locked(void) {
    if (g_trace_path[0] != '\0') return;

    const char* env_path = getenv("TFLITE_PATCH_TRACE_FILE");
    if (env_path && env_path[0] != '\0') {
        snprintf(g_trace_path, sizeof(g_trace_path), "%s", env_path);
        return;
    }

    snprintf(g_trace_path, sizeof(g_trace_path),
             "/tmp/tflite_patch_trace_%ld.log", (long)getpid());
}

void tracef(const char* tag, const char* fmt, ...) {
    pthread_mutex_lock(&g_trace_lock);
    ensure_trace_path_locked();

    int fd = open(g_trace_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd >= 0) {
        dprintf(fd, "[%.6f] [%s] ", now_realtime(), tag);

        va_list args;
        va_start(args, fmt);
        vdprintf(fd, fmt, args);
        va_end(args);

        dprintf(fd, "\n");
        close(fd);
    }

    pthread_mutex_unlock(&g_trace_lock);
}

void set_trace_output_path(const char* path) {
    pthread_mutex_lock(&g_trace_lock);
    if (path && path[0] != '\0') {
        snprintf(g_trace_path, sizeof(g_trace_path), "%s", path);
    }
    pthread_mutex_unlock(&g_trace_lock);
}

void set_tensor_trace_limit(int limit) {
    g_tensor_limit = limit;
}

void set_tensor_max_bytes(size_t max_bytes) {
    g_max_tensor_bytes = max_bytes;
}

void set_tensor_preview_values(int count) {
    g_preview_values = count < 0 ? 0 : count;
}

void set_tensor_name_logging(int enabled) {
    g_tensor_name_logging = enabled ? 1 : 0;
}

void set_trace_all_tensors(int enabled) {
    g_trace_all_tensors = enabled ? 1 : 0;
}

void set_target_subgraph(int subgraph_idx) {
    g_target_subgraph = subgraph_idx;
}

void set_fault_injection(int mode, float param, int tensor_idx) {
    pthread_mutex_lock(&g_trace_lock);
    g_fault_mode = mode;
    g_fault_param = param;
    g_fault_tensor_idx = tensor_idx;
    pthread_mutex_unlock(&g_trace_lock);
    tracef("FAULT", "configured mode=%d param=%.4f tensor=%d", mode, param, tensor_idx);
}

void clear_fault_injection(void) {
    pthread_mutex_lock(&g_trace_lock);
    g_fault_mode = FAULT_NONE;
    g_fault_param = 0.0f;
    g_fault_tensor_idx = -1;
    pthread_mutex_unlock(&g_trace_lock);
    tracef("FAULT", "cleared");
}

void set_custom_tensor_indices(const uint32_t* indices, uint32_t count) {
    pthread_mutex_lock(&g_trace_lock);
    if (!indices || count == 0) {
        g_custom_tensor_count = 0;
    } else {
        uint32_t n = count > MAX_CUSTOM_TENSORS ? MAX_CUSTOM_TENSORS : count;
        memcpy(g_custom_tensors, indices, n * sizeof(uint32_t));
        g_custom_tensor_count = n;
    }
    pthread_mutex_unlock(&g_trace_lock);
}

void clear_custom_tensor_indices(void) {
    pthread_mutex_lock(&g_trace_lock);
    g_custom_tensor_count = 0;
    pthread_mutex_unlock(&g_trace_lock);
}

void set_interpreter_layout(size_t subgraph_offset,
                            size_t tensor_array_offset,
                            size_t tensor_count_offset,
                            size_t tensor_size) {
    if (subgraph_offset > 0) g_interp_layout.interp_subgraph_loc = subgraph_offset;
    if (tensor_array_offset > 0) g_interp_layout.sg_tensor_array = tensor_array_offset;
    if (tensor_count_offset > 0) g_interp_layout.sg_tensor_count = tensor_count_offset;
    if (tensor_size > 0) g_interp_layout.tensor_size = tensor_size;
}

/* Internal profiler helper to record elapsed time */
void record_latency_sample(double elapsed_ms) {
    pthread_mutex_lock(&g_trace_lock);
    if (g_latency_count < MAX_LATENCY_SAMPLES) {
        g_latency_history[g_latency_count++] = elapsed_ms;
    } else {
        /* Ring replacement for long runs */
        g_latency_history[rand() % MAX_LATENCY_SAMPLES] = elapsed_ms;
    }
    pthread_mutex_unlock(&g_trace_lock);
}

void record_nan_inf_anomaly(void) {
    __sync_add_and_fetch(&g_nan_inf_counter, 1);
}

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

void get_latency_profile(const char* model_name, struct latency_profile* out) {
    (void)model_name;
    if (!out) return;
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock(&g_trace_lock);
    size_t count = g_latency_count;
    if (count == 0) {
        out->nan_inf_count = g_nan_inf_counter;
        pthread_mutex_unlock(&g_trace_lock);
        return;
    }

    double* copy = (double*)malloc(count * sizeof(double));
    if (!copy) {
        pthread_mutex_unlock(&g_trace_lock);
        return;
    }
    memcpy(copy, g_latency_history, count * sizeof(double));
    uint64_t nans = g_nan_inf_counter;
    pthread_mutex_unlock(&g_trace_lock);

    qsort(copy, count, sizeof(double), compare_doubles);

    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += copy[i];
    }

    out->total_calls = count;
    out->avg_ms = sum / (double)count;
    out->min_ms = copy[0];
    out->max_ms = copy[count - 1];

    size_t idx50 = (size_t)(count * 0.50);
    size_t idx90 = (size_t)(count * 0.90);
    size_t idx99 = (size_t)(count * 0.99);

    if (idx50 >= count) idx50 = count - 1;
    if (idx90 >= count) idx90 = count - 1;
    if (idx99 >= count) idx99 = count - 1;

    out->p50_ms = copy[idx50];
    out->p90_ms = copy[idx90];
    out->p99_ms = copy[idx99];
    out->nan_inf_count = nans;

    free(copy);
}

void reset_latency_profile(void) {
    pthread_mutex_lock(&g_trace_lock);
    g_latency_count = 0;
    g_nan_inf_counter = 0;
    pthread_mutex_unlock(&g_trace_lock);
}
