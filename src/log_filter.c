#include "tflite_patch.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static size_t (*real_fwrite)(const void*, size_t, size_t, FILE*) = NULL;

static int is_runtime_log(const char* buf, size_t len) {
    if (len < 5) return 0;
    return (buf[0] == 'I' || buf[0] == 'W' || buf[0] == 'E' || buf[0] == 'F')
        && (buf[1] >= '0' && buf[1] <= '9')
        && (buf[2] >= '0' && buf[2] <= '9')
        && (buf[3] >= '0' && buf[3] <= '9')
        && (buf[4] >= '0' && buf[4] <= '9');
}

TFLITE_PATCH_EXPORT size_t fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
    if (!real_fwrite) {
        real_fwrite = (size_t (*)(const void*, size_t, size_t, FILE*))
            dlsym(RTLD_NEXT, "fwrite");
    }

    size_t total = size * nmemb;
    const char* buf = (const char*)ptr;

    if (stream == stderr && total > 0 && is_runtime_log(buf, total)) {
        tracef("LIBLOG", "%.*s", (int)total, buf);
        return nmemb; /* Suppress noisy output from direct stderr */
    }

    return real_fwrite ? real_fwrite(ptr, size, nmemb, stream) : 0;
}
