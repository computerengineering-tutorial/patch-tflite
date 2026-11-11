#ifndef TFLITE_PATCH_TENSORS_H
#define TFLITE_PATCH_TENSORS_H

#include "tflite_patch.h"

void record_latency_sample(double elapsed_ms);
void record_nan_inf_anomaly(void);

void trace_subgraph_tensors(int call_id,
                            const struct model_snapshot* model,
                            const char* stage,
                            const char* kind,
                            const uint32_t* indices,
                            uint32_t count,
                            void* tensor_array,
                            uint64_t total_tensors,
                            double elapsed_ms);

void apply_tensor_overrides(const char* model_name,
                            int is_post,
                            void* tensor_array,
                            uint64_t total_tensors);

#endif /* TFLITE_PATCH_TENSORS_H */
