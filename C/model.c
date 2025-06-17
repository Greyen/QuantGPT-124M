#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct 
{
    int embedding_dims;
    int layers;
    int vocab_size;
    int seq_length ;
    int n_head;
    int batch_size;
} GPTconfig;

typedef struct Normalized{
    int input_dims;
    int output_dims;
    float* data;
}Normalized;

typedef struct MultiHead
{
    float* QKT;
    float* headOutput; 
}MultiHead;


typedef struct CasualAttention{
    float* q_proj;
    float* v_proj;
    float* k_proj;
    int n_head;
    MultiHead head;

}CasualAttention;

typedef struct Linear{
    int input_features;
    int output_features;
    float* data; 
}Linear;

typedef struct Bias{
    float* data;
}Bias;

typedef struct FFN{
    Linear lin_1; // (1,4)
    Bias b1; // 4
    Linear lin_2; // 4,1
    Bias b2; // 1
}FFN;

typedef struct DecoderBlock
{
    Normalized nor1;
    CasualAttention attn;
    Normalized nor2;
    FFN fn;
    

}DecoderBlock;

typedef struct Embedding{
    // int rows;
    // int col;
    float* data;
}Embedding;

typedef struct Positional{
    float* data;
}Positional;

typedef struct {
    Embedding* emb;
    Positional* pos;
    DecoderBlock* block;
    Normalized* LayerNorm;
    Linear* lin;
}GPT2;
typedef struct params{
    float* em_out;// can be put into with embedding will try once all done
    float* MHA_out;
}params;


typedef struct Tensor{
    // float batch[32];
    // float seq_len[128];
    // float dim[256];
    int data[2][4];
}Tensor;

#define MAX_TOKEN 50257

void print_3d_array(float res[2][4][128]) {
    for (int i = 0; i < 2; i++) {
        printf("res[%d]:\n", i);
        for (int j = 0; j < 4; j++) {
            printf("  res[%d][%d]: ", i, j);
            for (int k = 0; k < 128; k++) {
                printf("%f ", res[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
void* SHAforward(GPTconfig config ,params* param ){

}

void* MHAforward(GPTconfig config , GPT2* gpt2 ,params* param){
    
    for(int i=0;i<config.n_head;i++){
        SHAforward(config , param);
    }
}

void* forward(GPTconfig config , GPT2* gpt2 , Tensor* input,params* param){
    int B = config.batch_size;
    int T = config.seq_length;
    int C = config.embedding_dims;
    int dim0 = sizeof(input->data) / sizeof(input->data[0]);      // = 2
    int dim1 = sizeof(input->data[0]) / sizeof(input->data[0][0]);
    float res[2][4][128];//lets say hardcode
    param->em_out = (float*)malloc(config.batch_size*config.seq_length*config.embedding_dims*sizeof(float));
    for (int i=0;i<dim0;i++){
        for(int j=0;j<dim1;j++){
           float* em_ptr =  (gpt2->emb->data + (input->data[i][j]-1)*config.embedding_dims);
           float* pos_ptr = (gpt2->pos->data + j*config.embedding_dims);
           int length = config.embedding_dims;
           for (int k = 0; k < length; k++) {
                // printf("lookup table: %f ", *(em_ptr + k)); // or slice[i]
                // printf("postion:      %f ", *(pos_ptr + k));
                res[i][j][k] = *(em_ptr + k)+ *(pos_ptr+k);
                param->em_out[(i * config.seq_length * config.embedding_dims) + (j * config.embedding_dims) + k] = *(em_ptr + k)+ *(pos_ptr+k);
            }
        }  
    }
    print_3d_array(res);
    // haveto look at this solution in the end 
    // for (int i = 0; i < B; i++)
    // {
    //     for (int j = 0; j < T; j++)
    //     {
    //         for (int k = 0; k < C; k++)
    //         {
    //             param->em_out[i * T * C + j * C + k] = gpt2->emb->data[input->data[i * T + j] * C + k] + pos_emb[j * C + k];
    //         }
    //     }
    // }
    // Multi head // Decoder layer 
    for(int i=0;i<config.layers;i++){
        MHAforward(config,gpt2,param);
    }
    
}

void initialize_lookup(float* data, int vocab_size, int embedding_dims) {
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embedding_dims; j++) {
            int idx = i * embedding_dims + j;
            data[idx] = ((float)rand() / RAND_MAX); // example: random float between 0–1
        }
    }
}
float randn(float mean, float stddev) {
    // Generate a normally distributed value using Box-Muller transform
    float u1 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
    float u2 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}

void xavier_normal_init(float *tensor, int fan_in, int fan_out, int flag, float init_scale) {
    float stddev = sqrt(2.0 / (fan_in + fan_out)) * init_scale;
    float x = 1 / sqrt(2.0 * 8); // Adjusted handling for residual connections variance
    for (int i = 0; i < fan_in * fan_out; i++) {
        if (flag == 1) {
            tensor[i] = randn(0, stddev * x);
        } else {
            tensor[i] = randn(0, stddev);
        }
    }
}

int main(){
    GPTconfig config = {
        .embedding_dims = 128,
        .layers = 12,
        .vocab_size = 50257,
        .seq_length = 4,
        .n_head = 8,
        .batch_size=2,
    };
    // Embedding* lookup_table  = (Embedding*)malloc(sizeof(Embedding));
    // lookup_table->rows = config.vocab_size;
    // lookup_table->col =  config.embedding_dims;
    // lookup_table->data = malloc(lookup_table->rows*lookup_table->col*sizeof(float));
    // Positional* pos_enco  = (Positional*)malloc(sizeof(Positional));
    // pos_enco->rows = config.seq_length;
    // pos_enco->col =  config.embedding_dims;
    // pos_enco->data = malloc(pos_enco->rows*pos_enco->col*sizeof(float));
    // Positional* pos_enco  = (Positional*)malloc(sizeof(Positional));

    // // Block  Module
    // DecoderBlock* block =  (DecoderBlock*)malloc(sizeof(DecoderBlock));




    GPT2* gpt2 =  (GPT2*)malloc(sizeof(GPT2));
    // should put here pointers or defernce them to get the object right now going with the pointers
    // gpt2->emb = *lookup_table;
    // gpt2->pos_matrix = *pos_enco;
    params* param = (params*)malloc(sizeof(params));
    gpt2->emb = (Embedding*)malloc(sizeof(Embedding));
    gpt2->pos = (Positional*)malloc(sizeof(Positional));
    gpt2->emb->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    gpt2->pos->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    float init_scale = 0.02f;  // Reduced initial scale
    xavier_normal_init(gpt2->emb->data, config.vocab_size, config.embedding_dims, 0, init_scale);
    xavier_normal_init(gpt2->pos->data, config.seq_length, config.embedding_dims, 0, init_scale);
  

    // have to intialize with normal distribution
    // initialize_lookup(gpt2->emb->data, config.vocab_size, config.embedding_dims);
    // gpt2->pos_matrix = (Positional*)malloc(config.seq_length*config.embedding_dims*sizeof(Positional));
    // gpt2->block = (DecoderBlock*)malloc(config.layers*sizeof(DecoderBlock));
    // // block output should be in the normalized vector
    // gpt2->LayerNorm = (Normalized*)malloc(config.batch_size*config.seq_length*config.embedding_dims*sizeof(float));
    // gpt2->lin = (Linear*)malloc(config.embedding_dims*config.vocab_size*sizeof(float));
    // typedef struct Dim{
    //     float data[256];
    // }Dim;
    // Dim* dim = (Dim*)malloc(256*sizeof(float));
    // for(int i=0;i<256;i++){
    //     dim->data[i]=32.5f;
    //     printf("number: "%f\n",dim->data[i]);
    // }
    int count=0;
    Tensor* sin_Tensor = (Tensor*)malloc(sizeof(Tensor));
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
            sin_Tensor->data[i][j] = (int)rand()% MAX_TOKEN;
        }
    }
    forward(config,gpt2,sin_Tensor,param);
    
}


