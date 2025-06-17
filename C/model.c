#include <stdlib.h>
#include <stdio.h>

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

typedef struct CaualAttention{
    float* q;
    float* v;
    float* k;
    int head;
    

}CaualAttention;

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
    CaualAttention attn;
    FFN fn;
    Normalized nor2;

}DecoderBlock;

typedef struct Embedding{
    int rows;
    int col;
    float* data;
}Embedding;

typedef struct Positional{
    int rows;
    int col;
    float* data;
}Positional;

typedef struct {
    Embedding* emb;
    Positional* pos_matrix;
    DecoderBlock* block;
    Normalized* LayerNorm;
    Linear* lin;
}GPT2;

int main(){
    GPTconfig config = {
        .embedding_dims = 768,
        .layers = 12,
        .vocab_size = 50257,
        .seq_length = 1024,
        .n_head = 12,
        .batch_size=32,
    };
    Embedding* lookup_table  = (Embedding*)malloc(sizeof(Embedding));
    lookup_table->rows = config.vocab_size;
    lookup_table->col =  config.embedding_dims;
    lookup_table->data = malloc(lookup_table->rows*lookup_table->col*sizeof(float));
    Positional* pos_enco  = (Positional*)malloc(sizeof(Positional));
    pos_enco->rows = config.seq_length;
    pos_enco->col =  config.embedding_dims;
    pos_enco->data = malloc(pos_enco->rows*pos_enco->col*sizeof(float));
    Positional* pos_enco  = (Positional*)malloc(sizeof(Positional));

    // Block  Module
    DecoderBlock* block =  (DecoderBlock*)malloc(sizeof(DecoderBlock));




    GPT2* gpt2 =  (GPT2*)malloc(sizeof(GPT2));
    // should put here pointers or defernce them to get the object right now going with the pointers
    // gpt2->emb = *lookup_table;
    // gpt2->pos_matrix = *pos_enco;

    gpt2->emb = (Embedding*)malloc(config.vocab_size*config.embedding_dims*sizeof(Embedding));
    gpt2->pos_matrix = (Positional*)malloc(config.seq_length*config.embedding_dims*sizeof(Positional));
    gpt2->block = (DecoderBlock*)malloc(config.layers*sizeof(DecoderBlock));
    // block output should be in the normalized vector
    gpt2->LayerNorm = (Normalized*)malloc(config.batch_size*config.seq_length*config.embedding_dims*sizeof(float));
    gpt2->lin = (Linear*)malloc(config.embedding_dims*config.vocab_size*sizeof(float));
}


