#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define MOCK_EXPORT __attribute__((visibility("default"), noinline))
#else
#define MOCK_EXPORT
#endif

typedef struct MockTfLiteIntArray {
    int size;
    int data[8];
} MockTfLiteIntArray;

typedef struct MockTfLiteTensor {
    int type;                     /* 0x00: 1 = FLOAT32 */
    void* data;                   /* 0x08 */
    MockTfLiteIntArray* dims;     /* 0x10 */
    char padding1[0x10];          /* 0x18 - 0x27 */
    size_t bytes;                 /* 0x28 */
    char padding2[0x40];          /* 0x30 - 0x6F: Total 0x70 */
} MockTfLiteTensor;

typedef struct MockSubgraph {
    char padding1[0x08];          /* 0x00 */
    MockTfLiteTensor* tensors;    /* 0x08 */
    uint64_t tensor_count;        /* 0x10 */
} MockSubgraph;

typedef struct MockInterpreter {
    char padding[0x68];           /* 0x00 - 0x67 */
    MockSubgraph* subgraphs[1];   /* 0x68 */
} MockInterpreter;

typedef struct MockTfLiteInterpreterWrapper {
    MockInterpreter* impl;        /* offset 0 */
} MockTfLiteInterpreterWrapper;

static MockTfLiteIntArray g_default_dims = {
    .size = 2,
    .data = {1, 4}
};

/* Internal C++ style invoke function */
MOCK_EXPORT int mock_internal_invoke(MockInterpreter* interp) {
    __asm__ volatile ("nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;");
    if (!interp || !interp->subgraphs[0]) return -1;
    MockSubgraph* sg = interp->subgraphs[0];
    if (sg->tensor_count < 2) return -2;

    MockTfLiteTensor* in_tensor = &sg->tensors[0];
    MockTfLiteTensor* out_tensor = &sg->tensors[1];

    float* in_vals = (float*)in_tensor->data;
    float* out_vals = (float*)out_tensor->data;
    size_t count = in_tensor->bytes / sizeof(float);

    for (size_t i = 0; i < count; i++) {
        out_vals[i] = in_vals[i] * 2.0f;
    }
    return 0;
}

/* Standard TFLite C API Invoke */
MOCK_EXPORT int TfLiteInterpreterInvoke(MockTfLiteInterpreterWrapper* wrapper) {
    __asm__ volatile ("nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;");
    if (!wrapper || !wrapper->impl) return -1;
    return mock_internal_invoke(wrapper->impl);
}

MOCK_EXPORT MockTfLiteInterpreterWrapper* mock_tflite_create(int count) {
    if (count <= 0) count = 4;

    MockTfLiteInterpreterWrapper* wrapper = (MockTfLiteInterpreterWrapper*)calloc(1, sizeof(*wrapper));
    MockInterpreter* interp = (MockInterpreter*)calloc(1, sizeof(*interp));
    MockSubgraph* sg = (MockSubgraph*)calloc(1, sizeof(*sg));
    MockTfLiteTensor* tensors = (MockTfLiteTensor*)calloc(2, sizeof(MockTfLiteTensor));

    /* Tensor 0: Input float array */
    tensors[0].type = 1; /* FLOAT32 */
    tensors[0].dims = &g_default_dims;
    tensors[0].bytes = count * sizeof(float);
    tensors[0].data = malloc(tensors[0].bytes);
    float* in_vals = (float*)tensors[0].data;
    for (int i = 0; i < count; i++) {
        in_vals[i] = (float)(i + 1); /* [1.0, 2.0, 3.0, 4.0] */
    }

    /* Tensor 1: Output float array */
    tensors[1].type = 1; /* FLOAT32 */
    tensors[1].dims = &g_default_dims;
    tensors[1].bytes = count * sizeof(float);
    tensors[1].data = malloc(tensors[1].bytes);
    memset(tensors[1].data, 0, tensors[1].bytes);

    sg->tensors = tensors;
    sg->tensor_count = 2;

    interp->subgraphs[0] = sg;
    wrapper->impl = interp;

    return wrapper;
}

MOCK_EXPORT void mock_tflite_free(MockTfLiteInterpreterWrapper* wrapper) {
    if (!wrapper) return;
    if (wrapper->impl) {
        MockInterpreter* interp = wrapper->impl;
        if (interp->subgraphs[0]) {
            MockSubgraph* sg = interp->subgraphs[0];
            if (sg->tensors) {
                free(sg->tensors[0].data);
                free(sg->tensors[1].data);
                free(sg->tensors);
            }
            free(sg);
        }
        free(interp);
    }
    free(wrapper);
}

MOCK_EXPORT void mock_tflite_set_input(MockTfLiteInterpreterWrapper* wrapper, const float* data, int count) {
    if (!wrapper || !wrapper->impl || !data) return;
    MockSubgraph* sg = wrapper->impl->subgraphs[0];
    memcpy(sg->tensors[0].data, data, count * sizeof(float));
}

MOCK_EXPORT void mock_tflite_get_output(MockTfLiteInterpreterWrapper* wrapper, float* out, int count) {
    if (!wrapper || !wrapper->impl || !out) return;
    MockSubgraph* sg = wrapper->impl->subgraphs[0];
    memcpy(out, sg->tensors[1].data, count * sizeof(float));
}

MOCK_EXPORT size_t mock_tflite_get_tensor_bytes(MockTfLiteInterpreterWrapper* wrapper, int tensor_idx) {
    if (!wrapper || !wrapper->impl) return 0;
    MockSubgraph* sg = wrapper->impl->subgraphs[0];
    if (tensor_idx < 0 || (uint64_t)tensor_idx >= sg->tensor_count) return 0;
    return sg->tensors[tensor_idx].bytes;
}

static MockTfLiteIntArray g_resized_dims[2];

MOCK_EXPORT int TfLiteInterpreterResizeInputTensor(MockTfLiteInterpreterWrapper* wrapper,
                                                  int32_t input_index,
                                                  const int* input_dims,
                                                  int32_t input_dims_size) {
    if (!wrapper || !wrapper->impl || input_index != 0 || !input_dims || input_dims_size <= 0) {
        return -1;
    }
    MockSubgraph* sg = wrapper->impl->subgraphs[0];
    if (sg->tensor_count < 2) return -1;

    int total_elements = 1;
    g_resized_dims[0].size = input_dims_size;
    for (int i = 0; i < input_dims_size && i < 8; i++) {
        g_resized_dims[0].data[i] = input_dims[i];
        total_elements *= input_dims[i];
    }

    size_t new_bytes = total_elements * sizeof(float);

    /* Reallocate input tensor */
    sg->tensors[0].dims = &g_resized_dims[0];
    sg->tensors[0].bytes = new_bytes;
    sg->tensors[0].data = realloc(sg->tensors[0].data, new_bytes);

    /* Also match output tensor size */
    g_resized_dims[1] = g_resized_dims[0];
    sg->tensors[1].dims = &g_resized_dims[1];
    sg->tensors[1].bytes = new_bytes;
    sg->tensors[1].data = realloc(sg->tensors[1].data, new_bytes);
    memset(sg->tensors[1].data, 0, new_bytes);

    return 0;
}

MOCK_EXPORT int TfLiteInterpreterAllocateTensors(MockTfLiteInterpreterWrapper* wrapper) {
    if (!wrapper || !wrapper->impl) return -1;
    return 0;
}

MOCK_EXPORT int TfLiteInterpreterModifyGraphWithDelegate(MockTfLiteInterpreterWrapper* wrapper, void* delegate) {
    (void)wrapper;
    (void)delegate;
    return 0;
}

