#include "TCopyCPU.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <iostream>


TCopyCPU::TCopyCPU()
{

};

TCopyCPU::~TCopyCPU()
{

};


void TCopyCPU::process(            
    int64_t sparse_pointer_addr, // int64 *
    int64_t sparse_dim_0, // Gabor ID
    int64_t sparse_dim_1, // Gabor Parameter

    int64_t gabor_pointer_addr, // float32 *
    int64_t gabor_dim_0, // Gabor ID
    int64_t gabor_dim_1, // X
    int64_t gabor_dim_2, // Y

    int64_t output_pointer_addr, // float32 *
    int64_t output_dim_0, // Pattern ID
    int64_t output_dim_1, // X
    int64_t output_dim_2, // Y

    int64_t number_of_cpu_processes
){
    int64_t* sparse_pointer = (int64_t*)sparse_pointer_addr;
    float* gabor_pointer = (float*)gabor_pointer_addr;
    float* output_pointer = (float*)output_pointer_addr;

    // Sparse Matrix
    assert((sparse_pointer != nullptr));
    assert((sparse_dim_0 > 0));
    assert((sparse_dim_1 > 0));

    // I assume three parameters: Pattern ID, Type, X, Y
    assert((sparse_dim_1 == 4));

    // Gabor Matrix
    assert((gabor_pointer != nullptr));
    assert((gabor_dim_0 > 0));
    assert((gabor_dim_1 > 0));
    assert((gabor_dim_2 > 0));

    // Output Matrix
    assert((output_pointer != nullptr));
    assert((output_dim_0 > 0));
    assert((output_dim_1 > 0));


    // Cache data for the pointer calculations
    size_t sparse_dim_c0 = sparse_dim_1;

    size_t gabor_dim_c0 = gabor_dim_1 * gabor_dim_2;
    // size_t gabor_dim_c1 = gabor_dim_2;

    size_t output_dim_c0 = output_dim_1 * output_dim_2;
    size_t output_dim_c1 = output_dim_2;
    
    assert((number_of_cpu_processes > 0));

    omp_set_num_threads(number_of_cpu_processes);

    // DEBUG:
    // omp_set_num_threads(1);

#pragma omp parallel for
    for (size_t gabor_id = 0; gabor_id < sparse_dim_0; gabor_id++)
    {
        process_sub(gabor_id, 
                    sparse_pointer, 
                    sparse_dim_c0, 

                    gabor_pointer, 
                    gabor_dim_0, 
                    gabor_dim_1, 
                    gabor_dim_2,
                    gabor_dim_c0,

                    output_pointer,
                    output_dim_0, 
                    output_dim_1, 
                    output_dim_2, 
                    output_dim_c0,
                    output_dim_c1
        );
    }

    return;
};


void TCopyCPU::process_sub(
    size_t gabor_id,
    int64_t* sparse_pointer,
    size_t sparse_dim_c0,

    float* gabor_pointer,
    int64_t gabor_dim_0,
    int64_t gabor_dim_1, 
    int64_t gabor_dim_2,  
    size_t gabor_dim_c0,

    float* output_pointer,
    int64_t output_dim_0,
    int64_t output_dim_1,
    int64_t output_dim_2,
    size_t output_dim_c0,
    size_t output_dim_c1    
    
){
    int64_t* sparse_offset = sparse_pointer + gabor_id * sparse_dim_c0;
    // Extract the gabor parameter
    int64_t gabor_pattern_id = sparse_offset[0];
    int64_t gabor_type = sparse_offset[1];
    int64_t gabor_x = sparse_offset[2];
    int64_t gabor_y = sparse_offset[3];

    // Filter out non valid stimulus ids -- we don't do anything
    if ((gabor_pattern_id < 0) || (gabor_pattern_id >= output_dim_0)) {
        printf("Stimulus ID=%li outside range [0, %li]!\n",
            (long int)gabor_pattern_id, (long int)output_dim_0);
        return;
    }

    // Filter out non valid patch types -- we don't do anything
    if ((gabor_type < 0) || (gabor_type >= gabor_dim_0)) {
        printf("Patch ID=%li outside range [0, %li]!\n",
            (long int)gabor_type, (long int)gabor_dim_0);
        return;
    }

    // X position is too big -- we don't do anything
    if (gabor_x >= output_dim_1) {
        return;
    }

    // Y position is too big -- we don't do anything
    if (gabor_y >= output_dim_2){
        return;
    }


    // Get the offset to the gabor patch
    float* gabor_offset = gabor_pointer + gabor_type * gabor_dim_c0;

    // Get the offset to the output image with the id pattern_id
    float* output_offset = output_pointer + gabor_pattern_id * output_dim_c0;

    float* output_position_x = nullptr;
    int64_t gabor_y_start = gabor_y;

    for (int64_t g_x = 0; g_x < gabor_dim_1; g_x ++) {

        // Start at the first y (i.e. last dimension) position
        gabor_y = gabor_y_start;

        // We copy only if we are on the output canvas -- X dimension
        if ((gabor_x >= 0) && (gabor_x < output_dim_1)) {

            // Where is our x line in memory?
            output_position_x = output_offset + gabor_x * output_dim_c1;

            for (int64_t g_y = 0; g_y < gabor_dim_2; g_y ++) {

                // We copy only if we are on the output canvas -- Y dimension
                if ((gabor_y >= 0) && (gabor_y < output_dim_2)) {
                    output_position_x[gabor_y] += *gabor_offset;
                }
                gabor_offset++;
                gabor_y++;
            }
        }
        // We skip an x line
        else
        {
            gabor_offset += gabor_dim_2;
        }
        gabor_x++;
    }

    return;
};
