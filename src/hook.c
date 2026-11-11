#include "tflite_patch.h"

#include <stdlib.h>

__attribute__((constructor))
static void init_from_env(void) {
    const char* limit = getenv("TFLITE_PATCH_TENSOR_LIMIT");
    if (limit && limit[0] != '\0') {
        g_tensor_limit = atoi(limit);
    }

    const char* max_bytes = getenv("TFLITE_PATCH_MAX_TENSOR_BYTES");
    if (max_bytes && max_bytes[0] != '\0') {
        long value = atol(max_bytes);
        if (value > 0) g_max_tensor_bytes = (size_t)value;
    }

    const char* preview = getenv("TFLITE_PATCH_PREVIEW_VALUES");
    if (preview && preview[0] != '\0') {
        g_preview_values = atoi(preview);
        if (g_preview_values < 0) g_preview_values = 0;
    }

    const char* names = getenv("TFLITE_PATCH_TENSOR_NAMES");
    if (names && names[0] != '\0') {
        g_tensor_name_logging = atoi(names) != 0;
    }

    const char* dump_dir = getenv("TFLITE_PATCH_DUMP_DIR");
    if (dump_dir && dump_dir[0] != '\0') {
        const char* models = getenv("TFLITE_PATCH_DUMP_MODELS");
        enable_tensor_dump(dump_dir, models);
    }

    /* Auto-hook if specified in environment */
    const char* auto_symbol = getenv("TFLITE_PATCH_AUTO_HOOK_SYMBOL");
    if (auto_symbol && auto_symbol[0] != '\0') {
        const char* target_lib = getenv("TFLITE_PATCH_TARGET_LIB");
        install_invoke_hook_symbol(target_lib, auto_symbol);
    }
}
