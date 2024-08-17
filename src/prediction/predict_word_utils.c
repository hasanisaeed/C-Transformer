#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/matrix_utils.h"

double *flatten_matrix(double **matrix, int rows, int cols)
{
    double *flattened = (double *)malloc(rows * cols * sizeof(double));
    int index = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            flattened[index++] = matrix[i][j];
        }
    }

    return flattened;
}

double *linear_layer(double *flattened_vector, int flattened_size, int vocab_size)
{
    // Initialize the weights randomly for simplicity.
    double **weights = create_random_matrix(flattened_size, vocab_size);

    double *logits = (double *)malloc(vocab_size * sizeof(double));
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] = 0.0;
        for (int j = 0; j < flattened_size; j++)
        {
            logits[i] += flattened_vector[j] * weights[j][i];
        }
    }

    free_matrix(weights, flattened_size);

    return logits;
}

double *softmax_vector(double *logits, int size)
{
    double *probs = (double *)malloc(size * sizeof(double));
    double sum_exp = 0.0;

    for (int i = 0; i < size; i++)
    {
        probs[i] = exp(logits[i]);
        sum_exp += probs[i];
    }

    for (int i = 0; i < size; i++)
    {
        probs[i] /= sum_exp;
    }

    return probs;
}

int select_max_probability(double *probs, int vocab_size)
{
    int max_index = 0;
    double max_prob = probs[0];

    for (int i = 1; i < vocab_size; i++)
    {
        if (probs[i] > max_prob)
        {
            max_prob = probs[i];
            max_index = i;
        }
    }

    return max_index;
}
