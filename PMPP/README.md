# Learnings & Experminents

| Experiment               | CPU Time (sec) | GPU Compute Time (sec) | GPU Total Time (sec) | Key Insight                                                                    |
| ------------------------ | -------------- | ---------------------- | -------------------- | ------------------------------------------------------------------------------ |
| **Vector Addition**      |    0.005          | 0.000079               | 0.907                | Memory-bound operation; data transfer dominates GPU total time                 |
| **Grayscale Conversion** | 0.022          | 0.001                  | 0.19                 | GPU compute is ~20× faster, but memory allocation + transfer is the bottleneck |
| **Matmul v1 per thread** | 3.711          | 0.06                  | 0.15                 | GPU compute is ~25× faster even with memcpy, we took each thread as computing each unit in the final matrix|
| **Matmul v2 per row** | 70          | 0.093  [ ~v1]                 | 0.11                 | took a bigger matrix than experiment 3 ; GPU compute is ~700× faster even with memcpy, we took each row as computing each unit in the final matrix|
