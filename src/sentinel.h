#ifndef SENTINEL_H
#define SENTINEL_H

#include "tflite_patch.h"
#include <stddef.h>
#include <stdint.h>

void sentinel_init(void);
void sentinel_record_invocation(void);

#endif /* SENTINEL_H */

