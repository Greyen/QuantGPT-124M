#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

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
    int head_size;
} GPTconfig;

typedef struct Normalized{
    float* mean;
    float* grad_mean;
    float* std;
    float* grad_std;
}Normalized;

typedef struct MultiHead
{
    float* QKT;
    float* grad_QKT;
    float* headOutput;
    float* grad_headOutput; 
}MultiHead;

typedef struct Linear{
    // int input_features;
    // int output_features;
    float* data; 
    float* grad_data;
}Linear;

typedef struct Bias{
    float* data;
    float* grad_data;
}Bias;


// have to look at where dropout works
typedef struct Dropout{
    float* data;
    float* grad_data;
}Dropout;

typedef struct GELU{
    float* data;
    float* grad_data;
}GELU;

typedef struct FFN{
    Linear* lin_1; // (1,4)
    Bias* b1; // 4
    Linear* lin_2; // 4,1
    Bias* b2; // 1
    Dropout* drop;
    GELU* gelu; 
}FFN;

typedef struct Head{
    float* q_proj;
    float* grad_q_proj;
    float* v_proj;
    float* grad_v_proj;
    float* k_proj;
    float* grad_k_proj;
    float* q;
    float* grad_q;
    float* k;
    float* grad_k;
    float* v;
    float* grad_v;
    float* qk;
    float* grad_qk;
    float* Attn_out;
    float* grad_Attn_out;
    float* sc_Attn_out;
    float* grad_sc_Attn_out;
    float* softmax_out;
    float* grad_softmax_out;
}Head;

// typedef struct HeadBlock{
//     Head* data;
// }HeadBlock;
typedef struct CasualAttention{
    Head** hblock;

}CasualAttention;
typedef struct DecoderBlock
{
    params** param;
    Normalized* nor1;
    CasualAttention* attn;
    Normalized* nor2;
    FFN* ffn;

}DecoderBlock;

typedef struct Embedding{
    // int rows;
    // int col;
    float* data;
    float* grad_data;
}Embedding;

typedef struct Positional{
    float* data;
    float* grad_data;
}Positional;

typedef struct DecoderBlocks{
  DecoderBlock* data;
  DecoderBlock* gradata;
}DecoderBlocks;

typedef struct {
    Embedding* emb;
    
    Positional* pos;
    DecoderBlock** blocks;
    Normalized* fLayerNorm;
    Linear* flin;
}GPT2;
typedef struct params{
    float* em_out;// can be put into with embedding will try once all done
    float* grad_em_out;
    float* lNorm1_out;
    float* grad_lNorm1_out;
    float* lNorm2_out;
    float* grad_lNorm2_out;
    float* MHA_out;
    float* grad_MHA_out;
    float* FFN1_out; // b,t,4c
    float* grad_FFN1_out;
    float* gelu1_out;
    float* grad_gelu1_out;
    float* FFN2_out; // b,t,c
    float* grad_FFN2_out; 
    float* gelu2_out; // b,
    float* grad_gelu2_out;
    float* drop_out;
    float* grad_drop_out;
    float* res; // single decoder block output 
    float* grad_res;
    float* fNormres; // after final decoder
    float* grad_fNormres;
    float* final_res;
    float* grad_final_res;
    float* fi_softmax;
    float* grad_fi_softmax;
}params;

typedef struct GlobalParam{
    float* input;
    float* final_res;
}GlobalParam;




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

sum(int index ,float* input , GPTconfig config , int incr , float* tar ){
    int count=0;
    int ptr_len=index;
    int total  = config.batch_size*config.seq_length*config.embedding_dims;
    while(ptr_len<total){
        count+=input[ptr_len];
        ptr_len+=incr;
    }
    tar[index] = count/config.batch_size;
}

std(int index ,float* input , GPTconfig config , int incr , float* tar , float* mean){
    int count=0;
    int ptr_len=index;
    int total  = config.batch_size*config.seq_length*config.embedding_dims;
    while(ptr_len<total){
        count+=pow(input[ptr_len]-mean[ptr_len],2);
        ptr_len+=incr;
    }
    tar[index] = sqrtf(count/config.batch_size);
}

BatchNorm(int index, float* input  , float* out ,float * mean , float* std , GPTconfig config,int incr){
    int ptr_len=index;
    int total  = config.batch_size*config.seq_length*config.embedding_dims;
    while(ptr_len<total){
        out[ptr_len] = (input[ptr_len]-mean[ptr_len])/std[ptr_len];
        ptr_len+=incr;
    }
}

void* layerNorm1(float* input,GPTconfig config,DecoderBlock* block,float* output){
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        sum(i,input,config,config.seq_length*config.embedding_dims ,block->nor1->mean);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        std(i,input,config,config.seq_length*config.embedding_dims ,block->nor1->std,block->nor1->mean);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        BatchNorm(i,input,output,block->nor1->mean,block->nor1->std ,config,config.seq_length*config.embedding_dims);
    }
    
}

void* layerNorm2(float* input,GPTconfig config,DecoderBlock* block,float* output){
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        sum(i,input,config,config.seq_length*config.embedding_dims,output);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        std(i,input,config,config.seq_length*config.embedding_dims ,block->nor2->std,block->nor2->mean);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        BatchNorm(i,input,output,block->nor2->mean,block->nor2->std ,config,config.seq_length*config.embedding_dims);
    }
    
}

void* finalLayerNorm(float* input,GPTconfig config,float* output,Normalized* fnor){
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        sum(i,input,config,config.seq_length*config.embedding_dims ,fnor->mean);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        std(i,input,config,config.seq_length*config.embedding_dims ,fnor->std,fnor->mean);
    }
    for(int i=0;i<config.seq_length*config.embedding_dims;i++){
        BatchNorm(i,input,output,fnor->mean,fnor->std ,config,config.seq_length*config.embedding_dims);
    }   
}


// after attention when its done 
// void* layerNorm2(float* input,GPTconfig config,DecoderBlock* block){
//     for(int i=0;i<config.seq_length*config.embedding_dims;i++){
//         block->nor2->mean[i] = sum(i,input,config,config.seq_length*config.embedding_dims ,block->nor2->mean);
//     }
//     for(int i=0;i<config.seq_length*config.embedding_dims;i++){
//         block->nor2->std[i] = std(i,input,config,config.seq_length*config.embedding_dims ,block->nor2->mean);
//     }

// }

// have to reduce harcoding here 
void Matrixmul(float* input , float* proj,float* output,GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        assert(config.embedding_dims%config.n_head ==0 );
        for(int j=0;j<config.head_size;j++){
            int count=0;
            for(int k=0;k<config.embedding_dims;k++){
                count+=input[i*config.embedding_dims+k]*proj[k*config.embedding_dims+j];
            }
            output[i*config.embedding_dims+j] =count;
        }
    }

}
// why these doing because by mulioying by value Attn_out may have high values which can lead to one hot encoding affect while doing softmax;
// means token is seeing only single token for attention
void* ScaledAttn(float* input , float* output , GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length*config.head_size;i++){
        output[i] = input[i]/sqrt(config.head_size);
    }
}
void* Softmax(float* input , float* output, GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length*config.head_size;i++){
        output[i] = exp(input[i]);
    }
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        int count=0;
        for(int k=0;k<config.head_size;k++){
            count+=output[i*config.head_size+k];
        }
        for(int k=0;k<config.head_size;k++){
            output[i*config.head_size+k] = output[i*config.head_size+k]/count;
        }
    }
}
void* SingleHeadAttentionforwrd(Head* head , float* input,float* output,GPTconfig config){
    Matrixmul(input , head->q_proj,head->q,config);
    Matrixmul(input , head->k_proj,head->k,config);
    Matrixmul(input , head->v_proj,head->v,config);
    Matrixmul(head->q, head->k , head->qk,config);
    Matrixmul(head->qk, head->v , head->Attn_out,config);
    ScaledAttn(head->Attn_out , head->sc_Attn_out,config);
    Softmax(head->sc_Attn_out,head->softmax_out,config);
}
void* MultiHeadAttentionforwrd(float* input,float* output , GPTconfig config,DecoderBlock* block,params* param){
    // assert(config.embedding_dims%config.n_head == 0);
    // int head_size = config.embedding_dims/config.n_head ;
    
    for(int i=0;i<config.n_head;i++){
        SingleHeadAttentionforwrd(block->attn->hblock[i] ,input ,output,config);
        for(int j=0;j<config.batch_size*config.seq_length;j++){
            for(int k=0;k<config.head_size;k++){
                param->MHA_out[j*config.embedding_dims+k+i*config.batch_size+k] = block->attn->hblock[i]->softmax_out[i*config.head_size+k];
            }   
        }
    }


    //........final output be like B,T,C 

}

// have to look how Gaussian error Linear unit works

// so gelu is GELU(x)= x*m(where m is CDF OF normal )P(X<x) i.e from -infinity to x

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float gelu(float x) {
    return x * sigmoid(1.702f * x);  // Fast GELU approximation
}

float d_gelu(float x) {
    float z = 1.702f * x;
    float s = sigmoid(z);
    return s + x * s * (1.0f - s) * 1.702f;
}

// have to correct the bias as it should be brodcast have to look into it

void FFNmatmul1(float* input , float* weight , float* output,float* bias,GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        for(int j=0;j<4*config.embedding_dims;j++){
            int count=0;
            for(int k=0;k<config.embedding_dims;k++){
                count+=input[i*config.embedding_dims+k]*weight[k*config.embedding_dims+j];
            }
            // adding a single bias veector for same for every token have to look at it
            count += bias[j];
            output[i*config.embedding_dims+j] =count;
        }
    }
}

void FFNmatmul2(float* input , float* weight , float* output,float* bias,GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        for(int j=0;j<config.embedding_dims;j++){
            int count=0;
            for(int k=0;k<4*config.embedding_dims;k++){
                count+=input[i*config.embedding_dims+k]*weight[k*config.embedding_dims+j];
            }
            // adding a single bias veector for same for every token have to look at it
            count += bias[j];
            output[i*config.embedding_dims+j] =count;
        }
    }
}

void* finalMatmul(float* input , float* weight , float* output,GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        for(int j=0;j<config.vocab_size;j++){
            int count=0;
            for(int k=0;k<config.embedding_dims;k++){
                count+=input[i*config.embedding_dims+k]*weight[k*config.embedding_dims+j];
            }
            output[i*config.embedding_dims+j] =count;
        }
    }
}

void* Gelu(float* input ,float* output , GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length*config.embedding_dims;i++){
        output[i] = gelu(input[i]);
    }
}
void* finalMap(float* input ,float* output , GPTconfig config){
    for(int i=0;i<config.batch_size*config.seq_length*config.embedding_dims;i++){
        output[i] = input[i];
    }
}

void* Gelubackward(params* param ,GPTconfig config  ,float* back , float* local ,  float* grad_out){
    for(int i=0;i<config.batch_size*config.seq_length*config.embedding_dims;i++){
        param->grad_FFN2_out[i] = d_gelu(param->gelu2_out[i])*grad_out[i];
    }
    
}

void* FFNmatmul2backward(GPTconfig config , params* param , Linear* linear ,  float* grad_out){

  // orgiginal gelu1_out (B,T,4C)
  // gelu1_out.Traspose (B,4C,T) (matmul) grad_FFN2_out(B,T,C)
  // result or say linear->grad_data should be of shape 4C,C

  // this gradent will not flow backwards beacuse it doesn't have any children 
  int B = config.batch_size; // 2
  int C = config.embedding_dims; // 1
  int T = config.seq_length; // 2
  for(int b=0;b<B;b++){  // batches 
    for(int k=0;k<4*C;k++){ // give result for all tokens or say seq_length // which will give one batch result
        for(int i=0;i<C;i++){ // it will give result of token of a seq 
            for(int j=0;j<T;j++){
                // explain it later 
                linear->grad_data[k*C+i] += param->gelu1_out[b*T*4*C + j*4*C + k]*grad_out[j*C + b*C*T + i];
            }
        }
    }
  }


    //   param->grad_gelu1_out here it flow backward it will be of B,T,4C 
    //   linear->data from here it will flow i.e grad output (4C,C) -> iska transpose (C,4C)
    //   param->grad_FFN2_out (B,T,C)
    // grad_gelu1_out  = param->grad_FFN2_out(B,T,C)(matmul) (linear->data).transpose(C,4C) ..overall-> B,T,4C

  for(int b=0;b<B;b++){  // batches 
    for(int k=0;k<T;k++){ // give result for all tokens or say seq_length // which will give one batch result
        for(int i=0;i<4*C;i++){ // it will give result of token of a seq 
            for(int l=0;l<C;l++){
                // explain it later 
                param->grad_gelu1_out[b*4*C*T + k*4*C+i] = param->grad_FFN2_out[b*T*C +l+ k*C]*linear->data[i*C + l];
            }
        }
    }
  }

}

// Have to change the dimensions 
void* FFNmatmul1backward(GPTconfig config , params* param , Linear* linear ,  float* grad_out){

   // weights dimension here (C,4C)
  // original MHA_out  (B,T,C) // gradients will flow back here 
  // MHA_out.Traspose (B,C,T) (matmul) grad_FFN1_out(B,T,4C)

  // result or say linear->grad_data should be of shape C,4C

  // this gradent will not flow backwards beacuse it doesn't have any children 
  int B = config.batch_size; // 2
  int C = config.embedding_dims; // 1
  int T = config.seq_length; // 2

  for(int b=0;b<B;b++){  // batches 
    for(int k=0;k<4*C;k++){ // give result for all tokens or say seq_length // which will give one batch result
        for(int i=0;i<C;i++){ // it will give result of token of a seq 
            for(int j=0;j<T;j++){
                // explain it later 
                linear->grad_data[k*C+i] += param->MHA_out[b*T*4*C + j*4*C + k]*grad_out[j*C + b*C*T + i];
            }
        }
    }
  }


    //   param->grad_gelu1_out here it flow backward it will be of B,T,4C 
    //   linear->data from here it will flow i.e grad output (4C,C) -> iska transpose (C,4C)
    //   param->grad_FFN2_out (B,T,C)
    // grad_gelu1_out  = param->grad_FFN2_out(B,T,C)(matmul) (linear->data).transpose(C,4C) ..overall-> B,T,4C

  for(int b=0;b<B;b++){  // batches 
    for(int k=0;k<T;k++){ // give result for all tokens or say seq_length // which will give one batch result
        for(int i=0;i<4*C;i++){ // it will give result of token of a seq 
            for(int l=0;l<C;l++){
                // explain it later 
                param->grad_MHA_out[b*4*C*T + k*4*C+i] = param->grad_FFN1_out[b*T*C +l+ k*C]*linear->data[i*C + l];
            }
        }
    }
  }

}

void* FFNbackward(GPTconfig config , params* param ,DecoderBlock* block){
    //Gleu2 backword
    Gelubackward(param ,config ,param->grad_gelu2_out);

    // FFN 2 linear layer backword
    FFNmatmul2backward(config,param,block->ffn->lin_2,param->grad_FFN2_out);

    //Gleu1 backword
    Gelubackward(param ,config,param->grad_FFN1_out ,param->gelu1_out, param->grad_gelu1_out );

    // FFN 1 linear layer backword *IMP* have to check these dimensions
    FFNmatmul1backward(config,param,block->ffn->lin_1 ,param->grad_FFN1_out);
}

void* FFNforward(float* input ,float* output,GPTconfig config,DecoderBlock* block,params* param){
    FFNmatmul1(input , block->ffn->lin_1->data,param->FFN1_out,block->ffn->b1->data,config);
    Gelu(param->FFN1_out , param->gelu1_out,config);
    FFNmatmul2(param->gelu1_out , block->ffn->lin_2->data,param->FFN2_out,block->ffn->b2->data,config);
    Gelu(param->FFN2_out ,param->gelu2_out,config);
    finalMap(param->gelu2_out , param->res,config);
}

void* MHAbackward(GPTconfig config , GPT2* gpt2,DecoderBlock* block){
    
}

float CrossEntropyloss(float* pred , int* true ,GPTconfig config){
    // True be like =B,T
    // Pred be like = B,T,Vocab
    float loss=0.0f;
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        loss+= -logf(pred[i*config.vocab_size+true[i]]);  
    }
    return loss/(config.batch_size*config.seq_length);
}
void* DecoderBackward(GPTconfig config , GPT2* gpt2 ,params* param,DecoderBlock* block, int num){

    // FFN
    FFNbackward(param->lNorm2_out,param->res,config,block,param);

    // MHA 
    MHAbackward(config ,gpt2, block)
   

}
void* Decoderforward(GPTconfig config , GPT2* gpt2 ,params* param,DecoderBlock* block, int num){
    if(num==0){
        // Batch normalization , have to add epsilon
        layerNorm1(param->em_out,config,block,param->lNorm1_out);
    }
    else{
        // Batch normalization , have to add epsilon
        layernorm1(param->res,config,block,param->lNorm1_out);
    }
    CasualAttentionforwrd(param->lNorm2_out ,param->MHA_out,config,block,param);

    // Batch normalization
    layerNorm2(param->MHA_out,config,block,param->lNorm2_out);

    // FFN
    FFNforward(param->lNorm2_out,param->res,config,block,param);
}

// BackPropagation
void* backward(GPTconfig config ,GPT2* gpt2, params* param , int* true ){
    //LOSS 
    for(int i=0;i<config.batch_size*config.seq_length;i++){
        param->grad_fi_softmax[i*config.vocab_size + true[i]] = -1/param->fi_softmax[i*config.vocab_size + true[i]];
    }


    // normalized backward

    // Decoder backward
    for(int i=config.layers-1;i>=0;i--){
        DecoderBackward(config,gpt2,param,gpt2->blocks[i], i);
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
                res[i][j][k] = *(em_ptr + k)+ *(pos_ptr+k);
                param->em_out[(i * config.seq_length * config.embedding_dims) + (j * config.embedding_dims) + k] = *(em_ptr + k)+ *(pos_ptr+k);
            }
        }  
    }

    // uncomment to print it
    // print_3d_array(res);

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
        Decoderforward(config,gpt2,param,gpt2->blocks[i], i);
    }

    // final layer norm
    finalLayerNorm(param->res,config,param->fNormres,gpt2->fLayerNorm);

    // final linear mapping embedding to vocab size
    finalMatmul(param->fNormres, gpt2->flin->data , param->final_res,config);

    //final softmax
    Softmax(param->final_res , param->fi_softmax,config);

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

void gpt2_init(GPTconfig config , GPT2* gpt2){
    float init_scale = 0.02f; 
    params* param = (params*)malloc(sizeof(params));
    params** par_block = (params**)malloc(config.layers*sizeof(params*));
    gpt2->emb = (Embedding*)malloc(sizeof(Embedding));
    gpt2->pos = (Positional*)malloc(sizeof(Positional));
    gpt2->emb->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    gpt2->pos->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    // params* param = (params*)malloc(sizeof(params));
    DecoderBlock* decoder = (DecoderBlock*)malloc(sizeof(DecoderBlock));
    gpt2->blocks = (DecoderBlocks**)malloc(config.layers*sizeof(DecoderBlock*));
    for(int i=0;i<config.layers;i++){

        // layer norm 1 init
        gpt2->blocks[i]->nor1 = (Normalized*)malloc(sizeof(Normalized));
        gpt2->blocks[i]->nor1->mean = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));
        gpt2->blocks[i]->nor1->std = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));

        // Attention init
        gpt2->blocks[i]->attn = (CasualAttention*)malloc(sizeof(CasualAttention));

        // Heads init
        Head* head = (Head*)malloc(sizeof(Head));

        // have to check param how to send it 
        gpt2->blocks[i]->attn->hblock = (Head**)malloc(config.n_head*sizeof(Head*));
        assert(config.embedding_dims%config.n_head == 0);
        int head_size = config.embedding_dims/config.n_head ;
        param->MHA_out = (float*)malloc(config.batch_size*config.seq_length*config.embedding_dims*sizeof(float));
        for(int j=0;i<config.n_head;j++){
            gpt2->blocks[i]->attn->hblock[j]->q_proj = (float*)malloc(config.embedding_dims*head_size*sizeof(float));
            xavier_normal_init(gpt2->blocks[i]->attn->hblock[j]->q_proj, config.vocab_size, config.embedding_dims, 0, init_scale);
            gpt2->blocks[i]->attn->hblock[j]->k_proj = (float*)malloc(config.embedding_dims*head_size*sizeof(float));
            xavier_normal_init(gpt2->blocks[i]->attn->hblock[j]->k_proj, config.vocab_size, config.embedding_dims, 0, init_scale);
            gpt2->blocks[i]->attn->hblock[j]->v_proj = (float*)malloc(config.embedding_dims*head_size*sizeof(float));
            xavier_normal_init(gpt2->blocks[i]->attn->hblock[j]->v_proj, config.vocab_size, config.embedding_dims, 0, init_scale);

            gpt2->blocks[i]->attn->hblock[j]->q = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));
            gpt2->blocks[i]->attn->hblock[j]->k = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));
            gpt2->blocks[i]->attn->hblock[j]->v = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));

            gpt2->blocks[i]->attn->hblock[j]->qk       = (float*)malloc(config.batch_size*config.seq_length*config.seq_length*sizeof(float));
            gpt2->blocks[i]->attn->hblock[j]->Attn_out = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));
            gpt2->blocks[i]->attn->hblock[j]->sc_Attn_out = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));
            gpt2->blocks[i]->attn->hblock[j]->softmax_out = (float*)malloc(config.batch_size*config.seq_length*head_size*sizeof(float));
        }   


        // gpt2->blocks[i]->attn->q_proj = (float*)malloc(config.embedding_dims*config.embedding_dims*sizeof(float));
        // xavier_normal_init(gpt2->blocks[i]->attn->q_proj, config.vocab_size, config.embedding_dims, 0, init_scale);
        // gpt2->blocks[i]->attn->k_proj = (float*)malloc(config.embedding_dims*config.embedding_dims*sizeof(float));
        // xavier_normal_init(gpt2->blocks[i]->attn->k_proj, config.vocab_size, config.embedding_dims, 0, init_scale);
        // gpt2->blocks[i]->attn->v_proj = (float*)malloc(config.embedding_dims*config.embedding_dims*sizeof(float));
        // xavier_normal_init(gpt2->blocks[i]->attn->v_proj, config.vocab_size, config.embedding_dims, 0, init_scale);

        

        // (*(gpt2->blocks+i))->nor1 = (Normalized*)malloc(sizeof(Normalized)); can we do right here 

        // layer norm 2 init
        gpt2->blocks[i]->nor2 = (Normalized*)malloc(sizeof(Normalized));
        gpt2->blocks[i]->nor2->mean = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));
        gpt2->blocks[i]->nor2->std = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));

        // FFN init

        // 1 layer c,4c
        int C  = config.embedding_dims;
        gpt2->blocks[i]->ffn->lin_1 = (Linear*)malloc(sizeof(Linear));
        gpt2->blocks[i]->ffn->lin_1->data = (float*)malloc(C*4*C*sizeof(float));
        xavier_normal_init(gpt2->blocks[i]->ffn->lin_1->data, config.vocab_size, config.embedding_dims, 0, init_scale);
        
        // 1 layer bias
        gpt2->blocks[i]->ffn->b1 = (Linear*)malloc(sizeof(Linear));
        gpt2->blocks[i]->ffn->b1->data = (float*)malloc(4*C*sizeof(float));
        xavier_normal_init(gpt2->blocks[i]->ffn->b1->data, config.vocab_size, config.embedding_dims, 0, init_scale);

        // 2 layer 4c,c
        gpt2->blocks[i]->ffn->lin_2 = (Linear*)malloc(sizeof(Linear));
        gpt2->blocks[i]->ffn->lin_2->data = (float*)malloc(4*C*C*sizeof(float));
        xavier_normal_init(gpt2->blocks[i]->ffn->lin_2->data, config.vocab_size, config.embedding_dims, 0, init_scale);
        
        // 2 layer bias
        gpt2->blocks[i]->ffn->b2 = (Linear*)malloc(sizeof(Linear));
        gpt2->blocks[i]->ffn->b2->data = (float*)malloc(C*sizeof(float));
        xavier_normal_init(gpt2->blocks[i]->ffn->b2->data, config.vocab_size, config.embedding_dims, 0, init_scale);       
    }

    // finalLayerNorm
    gpt2->fLayerNorm = (Normalized*)malloc(sizeof(Normalized));
    gpt2->fLayerNorm->mean = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));
    gpt2->fLayerNorm->std = (float*)malloc(config.seq_length*config.embedding_dims*sizeof(float));
    

    //Final Linear Layer 
    gpt2->flin = (Linear*)malloc(sizeof(Linear));
    gpt2->flin->data = (float*)malloc(config.embedding_dims*config.vocab_size*sizeof(float));
    

}


int main(){
    GPTconfig config = {
        .embedding_dims = 128,
        .layers = 12,
        .vocab_size = 50257,
        .seq_length = 4,
        .n_head = 8,
        .batch_size=2,
        .head_size = config.embedding_dims/config.n_head
    };

    // Have to look what the fuck is this ---------------------------------------------------------------------------
    
    
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




    // Have to look what the fuck is this ------------------------------------------------------

    // GPT2* gpt2 =  (GPT2*)malloc(sizeof(GPT2));
    // // should put here pointers or defernce them to get the object right now going with the pointers
    // // gpt2->emb = *lookup_table;
    // // gpt2->pos_matrix = *pos_enco;
    params* param = (params*)malloc(sizeof(params));
    // gpt2->emb = (Embedding*)malloc(sizeof(Embedding));
    // gpt2->pos = (Positional*)malloc(sizeof(Positional));
    // gpt2->emb->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    // gpt2->pos->data = malloc(config.vocab_size * config.embedding_dims * sizeof(float));
    // float init_scale = 0.02f;  // Reduced initial scale
    // xavier_normal_init(gpt2->emb->data, config.vocab_size, config.embedding_dims, 0, init_scale);
    // xavier_normal_init(gpt2->pos->data, config.seq_length, config.embedding_dims, 0, init_scale);
  

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

    // --------------------------------------------------------------------------------------------------------------------
    int count=0;
    Tensor* sin_Tensor = (Tensor*)malloc(sizeof(Tensor));
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
            sin_Tensor->data[i][j] = (int)rand()% MAX_TOKEN;
        }
    }


    GPT2* gpt2 =  (GPT2*)malloc(sizeof(GPT2));

    // gpt inti
    gpt2_init(config,gpt2);

    // training iteration


    // batch's size // have to make data loader in c


    // forward
    forward(config,gpt2,sin_Tensor,param); // have to look at this param thing 

    float loss = CrossEntropyloss(param->final_res,);

    // make sure that every parameter's grad is zero
    
    // loss.backward()

    backward(config);
  
}


