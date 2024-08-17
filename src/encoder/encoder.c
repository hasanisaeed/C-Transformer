#include <stdio.h>
#include <stdlib.h>
#include "encoder.h"
#include "../common/embedding.h"
#include "../common/positional_encoding.h"
#include "../common/attention.h"
#include "../common/add_norm.h"
#include "../common/matrix_utils.h"
#include "../common/memory_utils.h"
#include "../common/text_utils.h"
#include "../feed_forward/feed_forward_network.h"

EncoderOutput *_malloc_encoder()
{
    EncoderOutput *result = (EncoderOutput *)malloc(sizeof(EncoderOutput));
    if (result == NULL)
    {
        fprintf(stderr, "Memory allocation failed for EncoderOutput\n");
        exit(1);
    }
    return result;
}

EncoderOutput *encoder(char *input)
{
    EncoderOutput *result = _malloc_encoder();

    int emv_rows, emv_cols;
    double **embedding_vector = create_embedding(input, &emv_rows, &emv_cols);

    // Positional Embedding Vector
    double **pe = calculate_pe_vector(emv_rows, emv_cols);

    // Concatenate Embedding and PE Vectors
    double **concate_pe_embv = add_matrices(pe, embedding_vector, emv_rows, emv_cols);

    print_matrix(concate_pe_embv, emv_rows, emv_cols, "Concatenate Embedding and PE Vectors");

    // Multi-Head Attention
    int mha_rows, mha_cols;
    double **mha_output = multi_head_attention(concate_pe_embv, emv_rows, emv_cols, &mha_rows, &mha_cols);

    // Add & Normalize (first)
    double **add_norm = add_and_normalize(mha_output, concate_pe_embv, mha_rows, mha_cols);

    // Feed Forward Network
    int ff_rows, ff_cols;
    double **ff_output = feed_forward(add_norm, mha_rows, mha_cols, &ff_rows, &ff_cols);

    // Add & Normalize (second)
    double **final_output = add_and_normalize(ff_output, add_norm, ff_rows, ff_cols);

    result->output = final_output;
    result->rows   = ff_rows;
    result->cols   = ff_cols;

    free_matrix(ff_output, ff_rows);        
    free_matrix(add_norm, mha_rows);        
    free_matrix(mha_output, mha_rows);      
    free_matrix(concate_pe_embv, emv_rows); 
    free_matrix(pe, emv_rows);              
    free_matrix(embedding_vector, emv_rows);

    return result;
}



