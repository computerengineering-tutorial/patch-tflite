#include "tensors.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct tensor_stats {
    int ok;
    size_t elements;
    double min;
    double max;
    double mean;
    int has_nan;
    int has_inf;
};

struct tensor_override_entry {
    int active;
    char model_name[MAX_MODEL_NAME];
    int is_post; /* 0 = PRE (modify inputs), 1 = POST (override outputs) */
    uint32_t tensor_idx;
    void* buffer;
    size_t byte_size;
};

static pthread_mutex_t g_dump_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_dump_enabled = 0;
static char g_dump_root[MAX_DUMP_PATH] = {0};
static char g_dump_models[MAX_DUMP_PATH * 2] = {0};

/* Zero-copy in-memory callback */
static pthread_mutex_t g_callback_lock = PTHREAD_MUTEX_INITIALIZER;
static tflite_tensor_callback_t g_tensor_callback = NULL;

/* Live tensor overrides */
static pthread_mutex_t g_override_lock = PTHREAD_MUTEX_INITIALIZER;
static struct tensor_override_entry g_overrides[MAX_TENSOR_OVERRIDES];
static int g_override_count = 0;

void set_tensor_callback(tflite_tensor_callback_t cb) {
    pthread_mutex_lock(&g_callback_lock);
    g_tensor_callback = cb;
    pthread_mutex_unlock(&g_callback_lock);
}

void clear_tensor_callback(void) {
    pthread_mutex_lock(&g_callback_lock);
    g_tensor_callback = NULL;
    pthread_mutex_unlock(&g_callback_lock);
}

int set_tensor_override(const char* model_name,
                        int is_post,
                        uint32_t tensor_idx,
                        const void* data,
                        size_t byte_size) {
    if (!data || byte_size == 0) return -1;

    pthread_mutex_lock(&g_override_lock);
    struct tensor_override_entry* entry = NULL;

    for (int i = 0; i < g_override_count; i++) {
        if (g_overrides[i].active &&
            g_overrides[i].is_post == is_post &&
            g_overrides[i].tensor_idx == tensor_idx &&
            ((!model_name && g_overrides[i].model_name[0] == '\0') ||
             (model_name && strcmp(g_overrides[i].model_name, model_name) == 0))) {
            entry = &g_overrides[i];
            break;
        }
    }

    if (!entry) {
        if (g_override_count >= MAX_TENSOR_OVERRIDES) {
            pthread_mutex_unlock(&g_override_lock);
            tracef("OVERRIDE", "registry_full cannot add override for tensor=%u", tensor_idx);
            return -1;
        }
        entry = &g_overrides[g_override_count++];
    }

    void* new_buf = realloc(entry->buffer, byte_size);
    if (!new_buf) {
        pthread_mutex_unlock(&g_override_lock);
        return -1;
    }

    entry->buffer = new_buf;
    memcpy(entry->buffer, data, byte_size);
    entry->byte_size = byte_size;
    entry->tensor_idx = tensor_idx;
    entry->is_post = is_post;
    entry->active = 1;
    if (model_name && model_name[0] != '\0') {
        snprintf(entry->model_name, sizeof(entry->model_name), "%s", model_name);
    } else {
        entry->model_name[0] = '\0';
    }

    pthread_mutex_unlock(&g_override_lock);

    tracef("OVERRIDE", "registered model=%s is_post=%d tensor=%u bytes=%zu",
           model_name ? model_name : "*", is_post, tensor_idx, byte_size);
    return 0;
}

void clear_tensor_overrides(void) {
    pthread_mutex_lock(&g_override_lock);
    for (int i = 0; i < g_override_count; i++) {
        if (g_overrides[i].buffer) {
            free(g_overrides[i].buffer);
            g_overrides[i].buffer = NULL;
        }
        g_overrides[i].active = 0;
    }
    g_override_count = 0;
    pthread_mutex_unlock(&g_override_lock);
    tracef("OVERRIDE", "all_cleared");
}

void apply_tensor_overrides(const char* model_name,
                            int is_post,
                            void* tensor_array,
                            uint64_t total_tensors) {
    if (!tensor_array || total_tensors == 0) return;

    pthread_mutex_lock(&g_override_lock);
    if (g_override_count == 0) {
        pthread_mutex_unlock(&g_override_lock);
        return;
    }

    for (int i = 0; i < g_override_count; i++) {
        struct tensor_override_entry* ov = &g_overrides[i];
        if (!ov->active || ov->is_post != is_post) continue;
        if (ov->model_name[0] != '\0' && model_name &&
            strstr(model_name, ov->model_name) == NULL) {
            continue;
        }

        uint32_t tidx = ov->tensor_idx;
        if ((uint64_t)tidx >= total_tensors) continue;

        void* tensor_ptr = (char*)tensor_array + ((size_t)tidx * g_interp_layout.tensor_size);
        if (!plausible_ptr(tensor_ptr)) continue;

        void* data = NULL;
        size_t bytes = 0;
        memcpy(&data, (char*)tensor_ptr + g_interp_layout.tensor_data, sizeof(data));
        memcpy(&bytes, (char*)tensor_ptr + g_interp_layout.tensor_bytes, sizeof(bytes));

        if (plausible_ptr(data) && bytes > 0) {
            size_t copy_bytes = ov->byte_size < bytes ? ov->byte_size : bytes;
            memcpy(data, ov->buffer, copy_bytes);
            tracef("OVERRIDE", "applied stage=%s tensor=%u bytes_copied=%zu",
                   is_post ? "POST" : "PRE", tidx, copy_bytes);
        }
    }

    pthread_mutex_unlock(&g_override_lock);
}

static int format_checked(char* out, size_t out_len, const char* fmt, ...) {
    if (!out || out_len == 0) return -1;

    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(out, out_len, fmt, args);
    va_end(args);

    if (n < 0 || (size_t)n >= out_len) {
        out[out_len - 1] = '\0';
        return -1;
    }
    return 0;
}

static const char* dtype_name(int dtype) {
    switch (dtype) {
        case TFLITE_TYPE_FLOAT32: return "f32";
        case TFLITE_TYPE_INT32: return "i32";
        case TFLITE_TYPE_UINT8: return "u8";
        case TFLITE_TYPE_INT64: return "i64";
        case TFLITE_TYPE_STRING: return "str";
        case TFLITE_TYPE_BOOL: return "bool";
        case TFLITE_TYPE_INT16: return "i16";
        case TFLITE_TYPE_COMPLEX64: return "complex64";
        case TFLITE_TYPE_INT8: return "i8";
        case TFLITE_TYPE_FLOAT16: return "f16";
        case TFLITE_TYPE_FLOAT64: return "f64";
        case TFLITE_TYPE_UINT32: return "u32";
        case TFLITE_TYPE_UINT16: return "u16";
        case TFLITE_TYPE_UINT64: return "u64";
        case TFLITE_TYPE_BFLOAT16: return "bf16";
        default: return "?";
    }
}

static void sanitize_component(const char* in, char* out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!in || in[0] == '\0') {
        snprintf(out, out_len, "unnamed");
        return;
    }

    size_t used = 0;
    for (size_t i = 0; in[i] && used + 1 < out_len; i++) {
        char c = in[i];
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '.' || c == '-' || c == '_') {
            out[used++] = c;
        } else {
            out[used++] = '_';
        }
    }
    out[used] = '\0';
}

static void json_escape(const char* in, char* out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!in) return;

    size_t used = 0;
    for (size_t i = 0; in[i] && used + 1 < out_len; i++) {
        unsigned char c = (unsigned char)in[i];
        if ((c == '"' || c == '\\') && used + 2 < out_len) {
            out[used++] = '\\';
            out[used++] = (char)c;
        } else if (c >= 0x20) {
            out[used++] = (char)c;
        }
    }
    out[used] = '\0';
}

static int mkdir_p(const char* path) {
    char tmp[MAX_DUMP_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return -1;

    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int write_file_atomic(const char* path, const void* data, size_t len) {
    char tmp_path[MAX_DUMP_PATH + 16];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp%ld", path, (long)getpid());

    int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return -1;

    const char* ptr = (const char*)data;
    size_t rem = len;
    while (rem > 0) {
        ssize_t written = write(fd, ptr, rem);
        if (written <= 0) {
            if (written < 0 && errno == EINTR) continue;
            close(fd);
            unlink(tmp_path);
            return -1;
        }
        ptr += written;
        rem -= (size_t)written;
    }

    if (close(fd) != 0) {
        unlink(tmp_path);
        return -1;
    }

    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return -1;
    }

    return 0;
}

int enable_tensor_dump(const char* root_dir, const char* model_paths_csv) {
    if (!root_dir || root_dir[0] == '\0') return -1;

    pthread_mutex_lock(&g_dump_lock);
    snprintf(g_dump_root, sizeof(g_dump_root), "%s", root_dir);
    if (model_paths_csv && model_paths_csv[0] != '\0') {
        snprintf(g_dump_models, sizeof(g_dump_models), "%s", model_paths_csv);
    } else {
        g_dump_models[0] = '\0';
    }
    g_dump_enabled = 1;

    int ok = mkdir_p(g_dump_root) == 0;
    pthread_mutex_unlock(&g_dump_lock);

    tracef("DUMP", "enabled root=%s filter=%s ok=%d",
           root_dir, model_paths_csv ? model_paths_csv : "*", ok);
    return ok ? 0 : -1;
}

void disable_tensor_dump(void) {
    pthread_mutex_lock(&g_dump_lock);
    g_dump_enabled = 0;
    g_dump_root[0] = '\0';
    g_dump_models[0] = '\0';
    pthread_mutex_unlock(&g_dump_lock);
    tracef("DUMP", "disabled");
}

static int model_matches_filter_locked(const char* model_name) {
    if (!g_dump_enabled) return 0;
    if (g_dump_models[0] == '\0') return 1;

    const char* p = g_dump_models;
    while (*p) {
        const char* next = strchr(p, ',');
        size_t len = next ? (size_t)(next - p) : strlen(p);
        if (len > 0) {
            char token[MAX_MODEL_NAME];
            if (len >= sizeof(token)) len = sizeof(token) - 1;
            memcpy(token, p, len);
            token[len] = '\0';

            if (strstr(model_name, token) != NULL) {
                return 1;
            }
        }
        if (!next) break;
        p = next + 1;
    }
    return 0;
}

static void compute_tensor_stats(int dtype, const void* data, size_t bytes,
                                 struct tensor_stats* stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!data || bytes == 0) return;
    if (g_max_tensor_bytes > 0 && bytes > g_max_tensor_bytes) return;

    switch (dtype) {
        case TFLITE_TYPE_FLOAT32: {
            const float* vals = (const float*)data;
            size_t n = bytes / sizeof(float);
            if (n == 0) return;
            double min_v = vals[0], max_v = vals[0], sum = 0.0;
            int nans = 0, infs = 0;
            for (size_t i = 0; i < n; i++) {
                float v = vals[i];
                if (isnan(v)) {
                    nans++;
                    record_nan_inf_anomaly();
                    continue;
                }
                if (isinf(v)) {
                    infs++;
                    record_nan_inf_anomaly();
                    continue;
                }
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }
            stats->ok = 1;
            stats->elements = n;
            stats->min = min_v;
            stats->max = max_v;
            stats->mean = (n > (size_t)(nans + infs)) ? (sum / (double)(n - nans - infs)) : 0.0;
            stats->has_nan = nans > 0;
            stats->has_inf = infs > 0;
            break;
        }
        case TFLITE_TYPE_INT32: {
            const int32_t* vals = (const int32_t*)data;
            size_t n = bytes / sizeof(int32_t);
            if (n == 0) return;
            double min_v = vals[0], max_v = vals[0], sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                int32_t v = vals[i];
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }
            stats->ok = 1;
            stats->elements = n;
            stats->min = min_v;
            stats->max = max_v;
            stats->mean = sum / (double)n;
            break;
        }
        case TFLITE_TYPE_UINT8: {
            const uint8_t* vals = (const uint8_t*)data;
            size_t n = bytes;
            if (n == 0) return;
            double min_v = vals[0], max_v = vals[0], sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                uint8_t v = vals[i];
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }
            stats->ok = 1;
            stats->elements = n;
            stats->min = min_v;
            stats->max = max_v;
            stats->mean = sum / (double)n;
            break;
        }
        case TFLITE_TYPE_INT8: {
            const int8_t* vals = (const int8_t*)data;
            size_t n = bytes;
            if (n == 0) return;
            double min_v = vals[0], max_v = vals[0], sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                int8_t v = vals[i];
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }
            stats->ok = 1;
            stats->elements = n;
            stats->min = min_v;
            stats->max = max_v;
            stats->mean = sum / (double)n;
            break;
        }
        case TFLITE_TYPE_INT64: {
            const int64_t* vals = (const int64_t*)data;
            size_t n = bytes / sizeof(int64_t);
            if (n == 0) return;
            double min_v = (double)vals[0], max_v = (double)vals[0], sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                int64_t v = vals[i];
                if ((double)v < min_v) min_v = (double)v;
                if ((double)v > max_v) max_v = (double)v;
                sum += (double)v;
            }
            stats->ok = 1;
            stats->elements = n;
            stats->min = min_v;
            stats->max = max_v;
            stats->mean = sum / (double)n;
            break;
        }
        default:
            break;
    }
}

static void format_tensor_preview(int dtype, const void* data, size_t bytes,
                                  int count, char* out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!data || bytes == 0 || count <= 0) return;

    size_t used = 0;
    out[used++] = '[';

    int max_vals = count > 16 ? 16 : count;

    switch (dtype) {
        case TFLITE_TYPE_FLOAT32: {
            const float* vals = (const float*)data;
            size_t n = bytes / sizeof(float);
            for (size_t i = 0; i < n && (int)i < max_vals && used + 24 < out_len; i++) {
                int w = snprintf(out + used, out_len - used, "%s%.4f", i ? "," : "", vals[i]);
                if (w > 0) used += (size_t)w;
            }
            break;
        }
        case TFLITE_TYPE_INT32: {
            const int32_t* vals = (const int32_t*)data;
            size_t n = bytes / sizeof(int32_t);
            for (size_t i = 0; i < n && (int)i < max_vals && used + 16 < out_len; i++) {
                int w = snprintf(out + used, out_len - used, "%s%d", i ? "," : "", vals[i]);
                if (w > 0) used += (size_t)w;
            }
            break;
        }
        case TFLITE_TYPE_UINT8: {
            const uint8_t* vals = (const uint8_t*)data;
            for (size_t i = 0; i < bytes && (int)i < max_vals && used + 8 < out_len; i++) {
                int w = snprintf(out + used, out_len - used, "%s%u", i ? "," : "", vals[i]);
                if (w > 0) used += (size_t)w;
            }
            break;
        }
        case TFLITE_TYPE_INT8: {
            const int8_t* vals = (const int8_t*)data;
            for (size_t i = 0; i < bytes && (int)i < max_vals && used + 8 < out_len; i++) {
                int w = snprintf(out + used, out_len - used, "%s%d", i ? "," : "", vals[i]);
                if (w > 0) used += (size_t)w;
            }
            break;
        }
        case TFLITE_TYPE_BOOL: {
            const uint8_t* vals = (const uint8_t*)data;
            for (size_t i = 0; i < bytes && (int)i < max_vals && used + 8 < out_len; i++) {
                int w = snprintf(out + used, out_len - used, "%s%s", i ? "," : "",
                                 vals[i] ? "T" : "F");
                if (w > 0) used += (size_t)w;
            }
            break;
        }
        default:
            break;
    }

    if (used < out_len - 1) {
        out[used++] = ']';
        out[used] = '\0';
    } else {
        out[out_len - 1] = '\0';
    }
}

static void format_tensor_shape(void* dims_ptr, char* out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';

    if (!plausible_ptr(dims_ptr)) {
        snprintf(out, out_len, "[]");
        return;
    }

    int32_t num_dims = 0;
    memcpy(&num_dims, dims_ptr, sizeof(num_dims));
    if (num_dims <= 0 || num_dims > 8) {
        snprintf(out, out_len, "[]");
        return;
    }

    int32_t dims[8] = {0};
    memcpy(dims, (char*)dims_ptr + sizeof(int32_t), sizeof(int32_t) * (size_t)num_dims);

    size_t used = 0;
    out[used++] = '[';
    for (int i = 0; i < num_dims && used + 16 < out_len; i++) {
        int w = snprintf(out + used, out_len - used, "%s%d", i ? "x" : "", dims[i]);
        if (w > 0) used += (size_t)w;
    }
    if (used < out_len - 1) {
        out[used++] = ']';
        out[used] = '\0';
    } else {
        out[out_len - 1] = '\0';
    }
}

static void maybe_dump_tensor(int call_id,
                              const struct model_snapshot* model,
                              const char* stage,
                              const char* kind,
                              uint32_t tensor_idx,
                              const char* tensor_name,
                              int dtype,
                              const char* shape_text,
                              const void* data,
                              size_t bytes,
                              const struct tensor_stats* stats) {
    pthread_mutex_lock(&g_dump_lock);
    if (!g_dump_enabled || !model_matches_filter_locked(model->name)) {
        pthread_mutex_unlock(&g_dump_lock);
        return;
    }

    char root[MAX_DUMP_PATH];
    snprintf(root, sizeof(root), "%s", g_dump_root);
    pthread_mutex_unlock(&g_dump_lock);

    char safe_model[MAX_MODEL_NAME];
    sanitize_component(model->name, safe_model, sizeof(safe_model));

    char model_dir[MAX_DUMP_PATH];
    if (format_checked(model_dir, sizeof(model_dir), "%s/%s_m%03d",
                       root, safe_model, model->id) != 0) {
        tracef("DUMP", "path_too_long dir=%s", model_dir);
        return;
    }
    if (mkdir_p(model_dir) != 0) return;

    char safe_tensor_name[MAX_TENSOR_NAME];
    sanitize_component(tensor_name, safe_tensor_name, sizeof(safe_tensor_name));

    char base_name[MAX_DUMP_PATH];
    if (format_checked(base_name, sizeof(base_name),
                       "call%06d_%s_%s_t%05u_%s",
                       call_id, stage, kind, tensor_idx, safe_tensor_name) != 0) {
        tracef("DUMP", "filename_too_long base=%s", base_name);
        return;
    }

    char bin_path[MAX_DUMP_PATH];
    char json_path[MAX_DUMP_PATH];
    if (format_checked(bin_path, sizeof(bin_path), "%s/%s.bin", model_dir, base_name) != 0 ||
        format_checked(json_path, sizeof(json_path), "%s/%s.json", model_dir, base_name) != 0) {
        tracef("DUMP", "full_path_too_long bin=%s json=%s", bin_path, json_path);
        return;
    }

    if (write_file_atomic(bin_path, data ? data : "", bytes) != 0) {
        tracef("DUMP", "write_bin_failed path=%s bytes=%zu errno=%d",
               bin_path, bytes, errno);
        return;
    }

    char json_text[2048];
    char escaped_model_name[MAX_MODEL_NAME * 2];
    char escaped_tensor_name[MAX_TENSOR_NAME * 2];
    json_escape(model->name, escaped_model_name, sizeof(escaped_model_name));
    json_escape(tensor_name ? tensor_name : "", escaped_tensor_name, sizeof(escaped_tensor_name));

    char stats_frag[256] = {0};
    if (stats && stats->ok) {
        snprintf(stats_frag, sizeof(stats_frag),
                 ",\n  \"elements\": %zu,\n  \"min\": %.6f,\n  \"max\": %.6f,\n  \"mean\": %.6f,\n  \"has_nan\": %s,\n  \"has_inf\": %s",
                 stats->elements, stats->min, stats->max, stats->mean,
                 stats->has_nan ? "true" : "false", stats->has_inf ? "true" : "false");
    }

    snprintf(json_text, sizeof(json_text),
             "{\n"
             "  \"call_id\": %d,\n"
             "  \"stage\": \"%s\",\n"
             "  \"kind\": \"%s\",\n"
             "  \"model_id\": %d,\n"
             "  \"subgraph_index\": %d,\n"
             "  \"model_name\": \"%s\",\n"
             "  \"tensor_index\": %u,\n"
             "  \"tensor_name\": \"%s\",\n"
             "  \"dtype\": \"%s\",\n"
             "  \"shape\": \"%s\",\n"
             "  \"bytes\": %zu,\n"
             "  \"bin\": \"%s.bin\"%s\n"
             "}\n",
             call_id, stage, kind, model->id, model->subgraph_index, escaped_model_name,
             tensor_idx, escaped_tensor_name, dtype_name(dtype),
             shape_text, bytes, base_name, stats_frag);

    if (write_file_atomic(json_path, json_text, strlen(json_text)) != 0) {
        tracef("DUMP", "write_json_failed path=%s errno=%d", json_path, errno);
        return;
    }

    tracef("DUMP", "saved call=%d %s %s t%u bytes=%zu file=%s.bin",
           call_id, stage, kind, tensor_idx, bytes, base_name);
}

void trace_subgraph_tensors(int call_id,
                            const struct model_snapshot* model,
                            const char* stage,
                            const char* kind,
                            const uint32_t* indices,
                            uint32_t count,
                            void* tensor_array,
                            uint64_t total_tensors,
                            double elapsed_ms) {
    if (!model || !tensor_array) return;

    int trace_limit = g_tensor_limit;
    if (trace_limit == 0 && !g_dump_enabled && !g_tensor_callback) return;

    /* Check if in-memory callback is registered */
    pthread_mutex_lock(&g_callback_lock);
    tflite_tensor_callback_t cb = g_tensor_callback;
    pthread_mutex_unlock(&g_callback_lock);

    for (uint32_t i = 0; i < count; i++) {
        if (trace_limit >= 0 && (int)i >= trace_limit && !g_dump_enabled && !cb) {
            tracef("TENSOR", "call=%d %s %s remaining=%u",
                   call_id, stage, kind, count - i);
            break;
        }

        uint32_t tensor_idx = indices ? indices[i] : i;
        if ((uint64_t)tensor_idx >= total_tensors) {
            tracef("TENSOR", "call=%d %s %s index_out_of_bounds idx=%u max=%llu",
                   call_id, stage, kind, tensor_idx,
                   (unsigned long long)total_tensors);
            continue;
        }

        void* tensor_ptr = (char*)tensor_array + ((size_t)tensor_idx * g_interp_layout.tensor_size);
        if (!plausible_ptr(tensor_ptr)) {
            tracef("TENSOR", "call=%d %s %s invalid_tensor_ptr idx=%u ptr=%p",
                   call_id, stage, kind, tensor_idx, tensor_ptr);
            continue;
        }

        int32_t dtype = 0;
        void* data = NULL;
        void* dims = NULL;
        size_t bytes = 0;

        memcpy(&dtype, (char*)tensor_ptr + g_interp_layout.tensor_type, sizeof(dtype));
        memcpy(&data, (char*)tensor_ptr + g_interp_layout.tensor_data, sizeof(data));
        memcpy(&dims, (char*)tensor_ptr + g_interp_layout.tensor_dims, sizeof(dims));
        memcpy(&bytes, (char*)tensor_ptr + g_interp_layout.tensor_bytes, sizeof(bytes));

        char shape_text[MAX_SHAPE_TEXT];
        format_tensor_shape(dims, shape_text, sizeof(shape_text));

        struct tensor_stats stats;
        compute_tensor_stats(dtype, data, bytes, &stats);

        if (stats.has_nan || stats.has_inf) {
            tracef("ANOMALY", "call=%d %s %s t%u has_nan=%d has_inf=%d",
                   call_id, stage, kind, tensor_idx, stats.has_nan, stats.has_inf);
        }

        char preview_text[128] = {0};
        if (g_preview_values > 0) {
            format_tensor_preview(dtype, data, bytes, g_preview_values,
                                  preview_text, sizeof(preview_text));
        }

        const char* tensor_name = NULL;
        if (indices) {
            if (strcmp(kind, "input") == 0 && i < model->input_count &&
                model->input_names[i][0] != '\0') {
                tensor_name = model->input_names[i];
            } else if (strcmp(kind, "output") == 0 && i < model->output_count &&
                       model->output_names[i][0] != '\0') {
                tensor_name = model->output_names[i];
            }
        }

        char name_frag[MAX_TENSOR_NAME + 16] = {0};
        if (g_tensor_name_logging && tensor_name && tensor_name[0] != '\0') {
            snprintf(name_frag, sizeof(name_frag), " name=%s", tensor_name);
        }

        if (trace_limit < 0 || (int)i < trace_limit) {
            if (stats.ok) {
                tracef("TENSOR",
                       "call=%d %s %s t%u type=%s shape=%s bytes=%zu min=%.4f max=%.4f mean=%.4f%s%s%s",
                       call_id, stage, kind, tensor_idx,
                       dtype_name(dtype), shape_text, bytes,
                       stats.min, stats.max, stats.mean,
                       preview_text[0] ? " preview=" : "",
                       preview_text,
                       name_frag);
            } else {
                tracef("TENSOR",
                       "call=%d %s %s t%u type=%s shape=%s bytes=%zu%s%s%s",
                       call_id, stage, kind, tensor_idx,
                       dtype_name(dtype), shape_text, bytes,
                       preview_text[0] ? " preview=" : "",
                       preview_text,
                       name_frag);
            }
        }

        /* Direct in-memory zero-copy callback */
        if (cb) {
            cb(call_id, stage, model->name, model->id, model->subgraph_index,
               tensor_idx, tensor_name ? tensor_name : "", dtype,
               shape_text, data, bytes, elapsed_ms);
        }

        maybe_dump_tensor(call_id, model, stage, kind, tensor_idx,
                          tensor_name ? tensor_name : "",
                          dtype, shape_text, data, bytes, &stats);
    }
}
