#include "attention.h"
#include "matrix_utils.h"
#include <math.h>
#include <stdio.h>

static double **compute_qk_transpose(double **Q, double **K, int rows, int cols, int *out_rows, int *out_cols)
{
    double **Kt = transpose_matrix(K, rows, cols);

    double **QKt = multiply_matrices(Q, Kt, rows, cols, cols, rows);
    *out_rows = rows;
    *out_cols = rows;

    free_matrix(Kt, cols);
    return QKt;
}

static double **apply_scaling_and_softmax(double **QKt, int rows)
{
    double **scaled_QKt = apply_function_on_matrix(QKt, rows, rows, sqrt);

    double **softmax_matrix = softmax(scaled_QKt, rows, rows);

    free_matrix(scaled_QKt, rows);
    return softmax_matrix;
}

static double **compute_attention_output(double **softmax_matrix, double **V, int rows, int cols)
{
    return multiply_matrices(softmax_matrix, V, rows, rows, rows, cols);
}

static void generate_qkv_matrices(int rows, int cols, double ***Q, double ***K, double ***V)
{
    *Q = create_random_matrix(rows, cols);
    *K = create_random_matrix(rows, cols);
    *V = create_random_matrix(rows, cols);
}

static void free_qkv_matrices(double **Q, double **K, double **V, int rows)
{
    free_matrix(Q, rows);
    free_matrix(K, rows);
    free_matrix(V, rows);
}

double **multi_head_attention(double **input_matrix, int rows, int cols, int *mha_rows, int *mha_cols)
{
    double **Q, **K, **V;
    generate_qkv_matrices(rows, cols, &Q, &K, &V);

    int qkt_rows, qkt_cols;
    double **QKt = compute_qk_transpose(Q, K, rows, cols, &qkt_rows, &qkt_cols);

    double **softmax_matrix = apply_scaling_and_softmax(QKt, qkt_rows);

    double **attention_output = compute_attention_output(softmax_matrix, V, rows, cols);

    free_qkv_matrices(Q, K, V, rows);
    free_matrix(QKt, qkt_rows);
    free_matrix(softmax_matrix, qkt_rows);

    *mha_rows = rows, *mha_cols = cols;

    print_matrix(attention_output, rows, cols, "Single Head Attention");

    return attention_output;
}

double **masked_multi_head_attention(double **input_matrix, int rows, int cols, int *mha_rows, int *mha_cols)
{ 
    double **Q = multiply_matrices(input_matrix, create_random_matrix(cols, cols), rows, cols, cols, cols);
    double **K = multiply_matrices(input_matrix, create_random_matrix(cols, cols), rows, cols, cols, cols);
    double **V = multiply_matrices(input_matrix, create_random_matrix(cols, cols), rows, cols, cols, cols);

    print_matrix(Q, rows, cols, "Q");
    print_matrix(K, rows, cols, "K");
    print_matrix(V, rows, cols, "V");

    double **QKt = compute_qk_transpose(Q, K, rows, cols, mha_rows, mha_cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            QKt[i][j] = QKt[i][j] / sqrt((double)cols);
        }
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = i + 1; j < rows; j++)
        {
            QKt[i][j] = -INFINITY;
        }
    }
    double **softmax_matrix = softmax(QKt, rows, cols);

    double **attention_output = compute_attention_output(softmax_matrix, V, rows, cols);

    free_qkv_matrices(Q, K, V, rows);
    free_matrix(QKt, rows);
    free_matrix(softmax_matrix, rows);

    print_matrix(attention_output, rows, cols, "Masked Multi-Head Attention");

    return attention_output;
}

double **multi_head_attention_with_encoder(double **decoder_input, int decoder_rows, int decoder_cols,
                                           double **encoder_output, int encoder_rows, int encoder_cols,
                                           int *mha_rows, int *mha_cols)
{
    double **Q = create_random_matrix(decoder_rows, decoder_cols);
    double **K = create_random_matrix(encoder_rows, encoder_cols);
    double **V = create_random_matrix(encoder_rows, encoder_cols);

    double **QKt = compute_qk_transpose(Q, K, decoder_rows, decoder_cols, mha_rows, mha_cols);
    double **softmax_matrix = apply_scaling_and_softmax(QKt, decoder_rows);
    double **attention_output = compute_attention_output(softmax_matrix, V, decoder_rows, encoder_cols);

    free_qkv_matrices(Q, K, V, decoder_rows);
    free_matrix(QKt, decoder_rows);
    free_matrix(softmax_matrix, decoder_rows);

    print_matrix(attention_output, decoder_rows, encoder_cols, "Multi-Head Attention with Encoder Output");

    return attention_output;
}
