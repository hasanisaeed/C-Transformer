#include <stdio.h>
#include <stdlib.h>
#include "decoder.h"
#include "../common/embedding.h"
#include "../common/positional_encoding.h"
#include "../common/attention.h"
#include "../common/add_norm.h"
#include "../common/matrix_utils.h"
#include "../common/memory_utils.h"
#include "../common/text_utils.h"
#include "../feed_forward/feed_forward_network.h"


DecoderOutput *_malloc_decoder()
{
    DecoderOutput *result = (DecoderOutput *)malloc(sizeof(DecoderOutput));
    if (result == NULL)
    {
        fprintf(stderr, "Memory allocation failed for DecoderOutput\n");
        exit(1);
    }
    return result;
}

DecoderOutput *decoder(char *input, double **encoder_output, int encoder_rows, int encoder_cols)
{
    DecoderOutput *result = _malloc_decoder();

    int emv_rows, emv_cols;
    double **embedding_vector = create_embedding(input, &emv_rows, &emv_cols);

    // Positional Encoding Vector
    double **pe = calculate_pe_vector(emv_rows, emv_cols);

    // Concatenate Embedding and PE Vectors
    double **concate_pe_embv = add_matrices(pe, embedding_vector, emv_rows, emv_cols);

    print_matrix(concate_pe_embv, emv_rows, emv_cols, "Concatenate Embedding and PE Vectors");

    // Masked Multi-Head Attention
    int mha_rows, mha_cols;
    double **masked_mha_output = masked_multi_head_attention(concate_pe_embv, emv_rows, emv_cols, &mha_rows, &mha_cols);

    // Add & Normalize (first)
    double **add_norm_1 = add_and_normalize(masked_mha_output, concate_pe_embv, mha_rows, mha_cols);

    // Multi-Head Attention with Encoder Output
    double **mha_with_encoder_output = multi_head_attention_with_encoder(add_norm_1, mha_rows, mha_cols, encoder_output, encoder_rows, encoder_cols, &mha_rows, &mha_cols);

    // Add & Normalize (second)
    double **add_norm_2 = add_and_normalize(mha_with_encoder_output, add_norm_1, mha_rows, mha_cols);

    // Feed Forward Network
    int ff_rows, ff_cols;
    double **ff_output = feed_forward(add_norm_2, mha_rows, mha_cols, &ff_rows, &ff_cols);

    // Add & Normalize (third)
    double **final_output = add_and_normalize(ff_output, add_norm_2, ff_rows, ff_cols);

    result->output = final_output;
    result->rows = ff_rows;
    result->cols = ff_cols;

    free_matrix(ff_output, ff_rows);
    free_matrix(add_norm_2, mha_rows);
    free_matrix(mha_with_encoder_output, mha_rows);
    free_matrix(add_norm_1, mha_rows);
    free_matrix(masked_mha_output, mha_rows);
    free_matrix(concate_pe_embv, emv_rows);

    return result;
}
