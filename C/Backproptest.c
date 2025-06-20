#include <stdio.h>
#include <stdlib.h>

int main() {
    // Configuration
    int B = 2; // batch_size
    int C = 1; // embedding_dims  
    int T = 2; // seq_length
    
    // Input arrays
    float gelu1_out[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float grad_out[4] = {17,18,19,20};
    
    // Output array - size should be 4*C = 4*1 = 4
    int grad_data_size = 4 * C;
    float* grad_data = (float*)calloc(grad_data_size, sizeof(float));
    
    printf("Input arrays:\n");
    printf("gelu1_out (size 16): ");
    for(int i = 0; i < 16; i++) printf("%.0f ", gelu1_out[i]);
    printf("\n");
    
    printf("grad_out (size 4): ");
    for(int i = 0; i < 4; i++) printf("%.0f ", grad_out[i]);
    printf("\n\n");
    
    // Corrected computation
    for(int b = 0; b < B; b++) {
        for(int k = 0; k < 4*C; k++) {
            for(int i = 0; i < C; i++) {
                for(int j = 0; j < T; j++) {
                    // Corrected indexing
                    int gelu_idx = b*T*C + j*C + i;  // Proper 3D indexing: [b][j][i]
                    // int grad_idx = b*T*C + j*C + i;  // Same indexing pattern
                    
                    grad_data[k*C + i] += gelu1_out[b*T*4*C + j*4*C + k] * grad_out[j*C + b*C*T + i];
                }
            }
        }
    }

    // for(int b=0;b<B;b++){
    //     for(int k=0;k<4*C;k++){
    //         for(int i=0;i<C;i++){
    //             for(int j=0;j<T;j++){
    //                 linear->grad_data[k*C+i] = param->gelu1_out[j*C + B*C*T +i]*grad_out[j*C+i];
    //             }
    //         }
    //     }
    // }
    
    printf("Computation details:\n");
    printf("For each k from 0 to %d:\n", 4*C-1);
    printf("  For each i from 0 to %d:\n", C-1);
    printf("    grad_data[%d] = sum over b,j of gelu1_out[b*%d+j*%d+i] * grad_out[b*%d+j*%d+i]\n", 
           0, T*C, C, T*C, C);
    printf("\n");
    
    // Show the computation step by step
    printf("Step-by-step computation:\n");
    for(int k = 0; k < 4*C; k++) {
        for(int i = 0; i < C; i++) {
            printf("grad_data[%d] = ", k*C + i);
            float sum = 0;
            for(int b = 0; b < B; b++) {
                for(int j = 0; j < T; j++) {
                    int gelu_idx = b*T*C + j*C + i;
                    int grad_idx = b*T*C + j*C + i;
                    float product = gelu1_out[gelu_idx] * grad_out[grad_idx];
                    sum += product;
                    printf("%.0f*%.0f", gelu1_out[gelu_idx], grad_out[grad_idx]);
                    if(!(b == B-1 && j == T-1)) printf(" + ");
                }
            }
            printf(" = %.0f\n", sum);
        }
    }
    
    printf("\nFinal result:\n");
    printf("linear->grad_data size: %d\n", grad_data_size);
    printf("linear->grad_data values: [");
    for(int i = 0; i < grad_data_size; i++) {
        printf("%.0f", grad_data[i]);
        if(i < grad_data_size-1) printf(", ");
    }
    printf("]\n");
    
    free(grad_data);
    return 0;
}