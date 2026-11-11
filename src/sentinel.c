#include "sentinel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

struct canary_slot {
    void* buffer;
    size_t offset;
    uint64_t expected_magic;
    int active;
};

static struct canary_slot g_canaries[MAX_CANARIES];
static int g_canary_count = 0;
static pthread_mutex_t g_sentinel_mutex = PTHREAD_MUTEX_INITIALIZER;

#define RSS_HISTORY_CAP 64
static uint64_t g_rss_history[RSS_HISTORY_CAP];
static int g_rss_count = 0;
static int g_leak_flag = 0;
static double g_leak_rate_kb = 0.0;

static void read_proc_statm(uint64_t* out_vsize, uint64_t* out_rss, uint64_t* out_data) {
    *out_vsize = 0;
    *out_rss = 0;
    *out_data = 0;

    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return;

    unsigned long size = 0, resident = 0, shared = 0, text = 0, lib = 0, data = 0, dt = 0;
    if (fscanf(f, "%lu %lu %lu %lu %lu %lu %lu", &size, &resident, &shared, &text, &lib, &data, &dt) >= 2) {
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) page_size = 4096;
        *out_vsize = (uint64_t)size * page_size;
        *out_rss = (uint64_t)resident * page_size;
        *out_data = (uint64_t)data * page_size;
    }
    fclose(f);
}

void sentinel_init(void) {
    pthread_mutex_lock(&g_sentinel_mutex);
    memset(g_canaries, 0, sizeof(g_canaries));
    g_canary_count = 0;
    g_rss_count = 0;
    g_leak_flag = 0;
    g_leak_rate_kb = 0.0;
    pthread_mutex_unlock(&g_sentinel_mutex);
}

void sentinel_record_invocation(void) {
    uint64_t vsize = 0, rss = 0, data = 0;
    read_proc_statm(&vsize, &rss, &data);

    pthread_mutex_lock(&g_sentinel_mutex);
    if (g_rss_count < RSS_HISTORY_CAP) {
        g_rss_history[g_rss_count++] = rss;
    } else {
        memmove(&g_rss_history[0], &g_rss_history[1], (RSS_HISTORY_CAP - 1) * sizeof(uint64_t));
        g_rss_history[RSS_HISTORY_CAP - 1] = rss;
    }

    if (g_rss_count >= 10) {
        /* Check trend over history */
        uint64_t first_rss = g_rss_history[0];
        uint64_t latest_rss = g_rss_history[g_rss_count - 1];
        if (latest_rss > first_rss) {
            uint64_t delta = latest_rss - first_rss;
            double rate = ((double)delta / 1024.0) / (double)(g_rss_count - 1);
            g_leak_rate_kb = rate;
            /* If consistently growing > 64 KB per invoke */
            if (rate > 64.0 && delta > (512 * 1024)) {
                g_leak_flag = 1;
            } else {
                g_leak_flag = 0;
            }
        } else {
            g_leak_rate_kb = 0.0;
            g_leak_flag = 0;
        }
    }
    pthread_mutex_unlock(&g_sentinel_mutex);
}

void sentinel_get_memory_info(struct memory_health_info* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));

    uint64_t vsize = 0, rss = 0, data = 0;
    read_proc_statm(&vsize, &rss, &data);

    pthread_mutex_lock(&g_sentinel_mutex);
    out->rss_bytes = rss;
    out->vsize_bytes = vsize;
    out->heap_bytes = data;
    out->leak_flag = g_leak_flag;
    out->leak_rate_kb = g_leak_rate_kb;
    out->canary_corruptions = sentinel_verify_canaries();
    pthread_mutex_unlock(&g_sentinel_mutex);
}

int sentinel_register_canary(void* buffer, size_t size) {
    if (!buffer || size < sizeof(uint64_t)) return -1;

    pthread_mutex_lock(&g_sentinel_mutex);
    for (int i = 0; i < MAX_CANARIES; i++) {
        if (!g_canaries[i].active) {
            g_canaries[i].buffer = buffer;
            g_canaries[i].offset = size - sizeof(uint64_t);
            g_canaries[i].expected_magic = SENTINEL_CANARY_MAGIC;
            g_canaries[i].active = 1;

            /* Write magic to buffer canary slot */
            uint64_t* canary_ptr = (uint64_t*)((char*)buffer + g_canaries[i].offset);
            *canary_ptr = SENTINEL_CANARY_MAGIC;

            if (i >= g_canary_count) g_canary_count = i + 1;
            pthread_mutex_unlock(&g_sentinel_mutex);
            return i;
        }
    }
    pthread_mutex_unlock(&g_sentinel_mutex);
    return -1;
}

int sentinel_verify_canaries(void) {
    int corruptions = 0;
    for (int i = 0; i < MAX_CANARIES; i++) {
        if (g_canaries[i].active && g_canaries[i].buffer) {
            uint64_t* canary_ptr = (uint64_t*)((char*)g_canaries[i].buffer + g_canaries[i].offset);
            if (*canary_ptr != g_canaries[i].expected_magic) {
                corruptions++;
            }
        }
    }
    return corruptions;
}

void sentinel_reset(void) {
    sentinel_init();
}

