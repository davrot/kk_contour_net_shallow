#ifndef TCOPYCPU
#define TCOPYCPU

#include <unistd.h>

#include <cctype>
#include <iostream>

class TCopyCPU
{
    public:
        TCopyCPU();
        ~TCopyCPU();

        void process(        
            int64_t sparse_pointer_addr, // int64 *
            int64_t sparse_dim_0, // Gabor ID
            int64_t sparse_dim_1, // Gabor Parameter

            int64_t garbor_pointer_addr, // float32 *
            int64_t garbor_dim_0, // Gabor ID
            int64_t garbor_dim_1, // X
            int64_t garbor_dim_2, // Y

            int64_t output_pointer_addr, // float32 *
            int64_t output_dim_0, // Pattern ID
            int64_t output_dim_1, // X
            int64_t output_dim_2, // Y

            int64_t number_of_cpu_processes
        );

    private:
        void process_sub(
            size_t gabor_id,
            int64_t* sparse_pointer,
            size_t sparse_dim_c0,
            float* garbor_pointer,
            int64_t garbor_dim_0,
            int64_t garbor_dim_1, 
            int64_t garbor_dim_2,  
            size_t garbor_dim_c0,
            float* output_pointer,
            int64_t output_dim_0,
            int64_t output_dim_1,
            int64_t output_dim_2,
            size_t output_dim_c0,
            size_t output_dim_c1  
        );
};
        
#endif /* TCOPYCPU */
