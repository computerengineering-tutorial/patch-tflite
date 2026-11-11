#include "models.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

struct model_entry {
    void* subgraph;
    uint64_t tensor_count;
    int id;
    int subgraph_index;
    char name[MAX_MODEL_NAME];
    uint32_t inputs[MAX_BOUNDARY_TENSORS];
    char input_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t input_count;
    uint32_t outputs[MAX_BOUNDARY_TENSORS];
    char output_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t output_count;
    int metadata_found;
    int ambiguous_model_path;
};

struct model_metadata {
    char name[MAX_MODEL_NAME];
    uint64_t tensor_count;
    uint32_t inputs[MAX_BOUNDARY_TENSORS];
    char input_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t input_count;
    uint32_t outputs[MAX_BOUNDARY_TENSORS];
    char output_names[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME];
    uint32_t output_count;
    uint64_t sequence;
};

static pthread_mutex_t g_model_lock = PTHREAD_MUTEX_INITIALIZER;
static struct model_entry g_models[MAX_MODELS];
static int g_model_count = 0;
static struct model_metadata g_model_metadata[MAX_MODEL_METADATA];
static int g_model_metadata_count = 0;
static uint64_t g_model_metadata_sequence = 0;

static void copy_indices(uint32_t* dst, const uint32_t* src, uint32_t count) {
    if (!dst || !src || count == 0) return;
    memcpy(dst, src, sizeof(uint32_t) * count);
}

static void copy_names(char dst[MAX_BOUNDARY_TENSORS][MAX_TENSOR_NAME],
                       const char* const* src, uint32_t count) {
    if (!dst || !src || count == 0) return;
    for (uint32_t i = 0; i < count; i++) {
        if (src[i] && src[i][0] != '\0') {
            snprintf(dst[i], MAX_TENSOR_NAME, "%s", src[i]);
        }
    }
}

int register_model_metadata_with_names(const char* path, uint64_t tensor_count,
                                       const uint32_t* inputs,
                                       uint32_t input_count,
                                       const char* const* input_names,
                                       const uint32_t* outputs,
                                       uint32_t output_count,
                                       const char* const* output_names) {
    if (!path || path[0] == '\0') return -1;
    if (input_count > MAX_BOUNDARY_TENSORS || output_count > MAX_BOUNDARY_TENSORS) {
        tracef("MODEL",
               "metadata_rejected name=%s tensor_count=%llu inputs=%u outputs=%u max=%u",
               path, (unsigned long long)tensor_count,
               input_count, output_count, (unsigned)MAX_BOUNDARY_TENSORS);
        return -1;
    }

    pthread_mutex_lock(&g_model_lock);

    struct model_metadata* meta = NULL;
    for (int i = 0; i < g_model_metadata_count; i++) {
        if (strncmp(g_model_metadata[i].name, path, MAX_MODEL_NAME) == 0) {
            meta = &g_model_metadata[i];
            break;
        }
    }

    if (!meta) {
        if (g_model_metadata_count >= MAX_MODEL_METADATA) {
            pthread_mutex_unlock(&g_model_lock);
            tracef("MODEL", "metadata_rejected name=%s reason=registry_full", path);
            return -1;
        }
        meta = &g_model_metadata[g_model_metadata_count++];
    }

    memset(meta, 0, sizeof(*meta));
    snprintf(meta->name, sizeof(meta->name), "%s", path);
    meta->tensor_count = tensor_count;
    meta->input_count = input_count;
    meta->output_count = output_count;
    copy_indices(meta->inputs, inputs, input_count);
    copy_indices(meta->outputs, outputs, output_count);
    copy_names(meta->input_names, input_names, input_count);
    copy_names(meta->output_names, output_names, output_count);
    meta->sequence = ++g_model_metadata_sequence;

    pthread_mutex_unlock(&g_model_lock);

    tracef("MODEL",
           "metadata_registered name=%s tensor_count=%llu inputs=%u outputs=%u",
           path, (unsigned long long)tensor_count, input_count, output_count);
    return 0;
}

int register_model_metadata(const char* path, uint64_t tensor_count,
                            const uint32_t* inputs, uint32_t input_count,
                            const uint32_t* outputs, uint32_t output_count) {
    return register_model_metadata_with_names(path, tensor_count,
                                              inputs, input_count, NULL,
                                              outputs, output_count, NULL);
}

void clear_model_metadata(void) {
    pthread_mutex_lock(&g_model_lock);
    g_model_count = 0;
    g_model_metadata_count = 0;
    g_model_metadata_sequence = 0;
    pthread_mutex_unlock(&g_model_lock);
    tracef("MODEL", "registry_cleared");
}

static const struct model_metadata* find_metadata_locked(uint64_t tensor_count,
                                                         int* ambiguous) {
    const struct model_metadata* match = NULL;
    int matches = 0;

    for (int i = 0; i < g_model_metadata_count; i++) {
        if (g_model_metadata[i].tensor_count == tensor_count) {
            match = &g_model_metadata[i];
            matches++;
        }
    }

    if (ambiguous) {
        *ambiguous = (matches > 1);
    }
    return match;
}

static void apply_metadata(struct model_entry* entry,
                           const struct model_metadata* meta,
                           int ambiguous) {
    entry->metadata_found = (meta != NULL);
    entry->ambiguous_model_path = ambiguous;

    if (meta) {
        snprintf(entry->name, sizeof(entry->name), "%s", meta->name);
        entry->input_count = meta->input_count;
        entry->output_count = meta->output_count;
        copy_indices(entry->inputs, meta->inputs, meta->input_count);
        copy_indices(entry->outputs, meta->outputs, meta->output_count);
        for (uint32_t i = 0; i < meta->input_count; i++) {
            snprintf(entry->input_names[i], sizeof(entry->input_names[i]), "%s",
                     meta->input_names[i]);
        }
        for (uint32_t i = 0; i < meta->output_count; i++) {
            snprintf(entry->output_names[i], sizeof(entry->output_names[i]), "%s",
                     meta->output_names[i]);
        }
    } else {
        snprintf(entry->name, sizeof(entry->name), "unnamed_tc%llu",
                 (unsigned long long)entry->tensor_count);
        entry->input_count = 0;
        entry->output_count = 0;
        memset(entry->inputs, 0, sizeof(entry->inputs));
        memset(entry->outputs, 0, sizeof(entry->outputs));
        memset(entry->input_names, 0, sizeof(entry->input_names));
        memset(entry->output_names, 0, sizeof(entry->output_names));
    }
}

static void fill_snapshot(const struct model_entry* entry,
                          struct model_snapshot* snapshot,
                          int is_new,
                          int tensor_count_changed) {
    if (!snapshot) return;

    snapshot->id = entry->id;
    snapshot->subgraph_index = entry->subgraph_index;
    snapshot->tensor_count = entry->tensor_count;
    snprintf(snapshot->name, sizeof(snapshot->name), "%s", entry->name);
    snapshot->input_count = entry->input_count;
    snapshot->output_count = entry->output_count;
    copy_indices(snapshot->inputs, entry->inputs, entry->input_count);
    copy_indices(snapshot->outputs, entry->outputs, entry->output_count);
    for (uint32_t i = 0; i < entry->input_count; i++) {
        snprintf(snapshot->input_names[i], sizeof(snapshot->input_names[i]), "%s",
                 entry->input_names[i]);
    }
    for (uint32_t i = 0; i < entry->output_count; i++) {
        snprintf(snapshot->output_names[i], sizeof(snapshot->output_names[i]), "%s",
                 entry->output_names[i]);
    }
    snapshot->metadata_found = entry->metadata_found;
    snapshot->ambiguous_model_path = entry->ambiguous_model_path;
    snapshot->is_new = is_new;
    snapshot->tensor_count_changed = tensor_count_changed;
}

int model_snapshot_for(void* subgraph, uint64_t tensor_count, int subgraph_idx,
                       struct model_snapshot* snapshot) {
    if (!subgraph || !snapshot) return 0;

    pthread_mutex_lock(&g_model_lock);

    struct model_entry* entry = NULL;
    int is_new = 0;
    int tensor_count_changed = 0;

    for (int i = 0; i < g_model_count; i++) {
        if (g_models[i].subgraph == subgraph) {
            entry = &g_models[i];
            break;
        }
    }

    if (!entry) {
        if (g_model_count >= MAX_MODELS) {
            pthread_mutex_unlock(&g_model_lock);
            return 0;
        }
        entry = &g_models[g_model_count++];
        memset(entry, 0, sizeof(*entry));
        entry->subgraph = subgraph;
        entry->id = g_model_count;
        entry->subgraph_index = subgraph_idx;
        entry->tensor_count = tensor_count;
        is_new = 1;

        int ambiguous = 0;
        const struct model_metadata* meta =
            find_metadata_locked(tensor_count, &ambiguous);
        apply_metadata(entry, meta, ambiguous);
    } else if (entry->tensor_count != tensor_count) {
        entry->tensor_count = tensor_count;
        entry->subgraph_index = subgraph_idx;
        tensor_count_changed = 1;

        int ambiguous = 0;
        const struct model_metadata* meta =
            find_metadata_locked(tensor_count, &ambiguous);
        apply_metadata(entry, meta, ambiguous);
    }

    fill_snapshot(entry, snapshot, is_new, tensor_count_changed);

    pthread_mutex_unlock(&g_model_lock);
    return entry->id;
}
