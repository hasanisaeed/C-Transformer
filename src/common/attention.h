#ifndef ATTENTION_H
#define ATTENTION_H

double **multi_head_attention(double **input_matrix, int rows, int cols, int *mha_rows, int *mha_cols);
double **masked_multi_head_attention(double **input_matrix, int rows, int cols, int *mha_rows, int *mha_cols);
double **multi_head_attention_with_encoder(double **decoder_input, int decoder_rows, int decoder_cols,
                                           double **encoder_output, int encoder_rows, int encoder_cols,
                                           int *mha_rows, int *mha_cols);
#endif // ATTENTION_H
