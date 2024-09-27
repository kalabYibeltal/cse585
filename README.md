# cse585
course project

Rethinking batching for CPU inference servers

GPU-poor companies that want to integrate LLMs into their product may consider CPU inference; most likely batch inference in an offline setting (real time serving on CPUs would be too slow). Existing works typically assume GPUs, which are extremely parallel hardware; for a compute instruction, it needs enough data to saturate all its compute units. That’s why batching multiple requests together was a no-brainer efficiency gain. However, what about CPU inference? AArch64 has 32 Neon registers and each is only 128 bits wide (four double-precision floating point numbers), providing different forms/degrees of parallelism compared to GPUs. Given this, is it better to process multiple requests in parallel on CPUs, or is it better to just do one at a time since one request is enough to saturate all vector registers? If parallelizing does provide some gains in makespan, what is the best way to parallelize?
