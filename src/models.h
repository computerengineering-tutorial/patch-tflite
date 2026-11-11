#ifndef TFLITE_PATCH_MODELS_H
#define TFLITE_PATCH_MODELS_H

#include "tflite_patch.h"

int model_snapshot_for(void* subgraph, uint64_t tensor_count, int subgraph_idx,
                       struct model_snapshot* snapshot);

#endif /* TFLITE_PATCH_MODELS_H */
