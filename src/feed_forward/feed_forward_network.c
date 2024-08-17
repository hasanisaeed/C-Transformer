#include <stdio.h>
#include <stdlib.h>
#include "feed_forward_network.h"
#include "../common/matrix_utils.h"
#include "../common/text_utils.h"

FeedForwardNetwork *create_feed_forward_network(int layer_sizes[LAYERS + 1])
{
    FeedForwardNetwork *network = (FeedForwardNetwork *)malloc(sizeof(FeedForwardNetwork));

    for (size_t i = 0; i < LAYERS; i++)
    {
        network->layer_sizes[i] = layer_sizes[i];
    }

    network->weights = (double ***)malloc(LAYERS * sizeof(double **));
    network->biases = (double ***)malloc(LAYERS * sizeof(double **));

    for (size_t i = 0; i < LAYERS; i++)
    {
        network->weights[i] = create_random_matrix(layer_sizes[i], layer_sizes[i + 1]);
        network->biases[i] = create_random_matrix(1, layer_sizes[i + 1]);
    }

    return network;
}

void free_feed_forward_network(FeedForwardNetwork *network)
{
    for (int l = 0; l < LAYERS; l++)
    {
        for (int i = 0; i < network->layer_sizes[l]; i++)
        {
            free(network->weights[l][i]);
        }
        free(network->weights[l]);
        free(network->biases[l]);
    }
    free(network->weights);
    free(network->biases);
    free(network);
}

double **add_bias(double **result, double **bias, int rows, int cols)
{
    double **tmp = create_2d_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            tmp[i][j] = result[i][j] + bias[0][j];
        }
    }

    return tmp;
}

// ReLU activation function
double **relu(double **matrix, int rows, int cols)
{
    double **tmp = create_2d_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (matrix[i][j] < 0)
            {
                tmp[i][j] = 0;
            }
            else
            {
                tmp[i][j] = matrix[i][j];
            }
        }
    }

    return tmp;
}

double **feed_forward(double **input, int rows, int cols, int *out_rows, int *out_cols)
{
    int layer_sizes[LAYERS + 1] = {cols, WEIGHT_MATRIX_COLUMNS};
    FeedForwardNetwork *network = create_feed_forward_network(layer_sizes);

    double **result = input;

    for (size_t l = 0; l < LAYERS; l++)
    {
        result = multiply_matrices(result, network->weights[l], rows, cols, layer_sizes[l], layer_sizes[l + 1]);
        result = add_bias(result, network->biases[l], rows, layer_sizes[l + 1]);
        result = relu(result, rows, layer_sizes[l + 1]);
        cols = layer_sizes[l + 1];
    }

    *out_rows = rows;
    *out_cols = cols;

    print_matrix(result, rows, cols, "Feed Forward Output");

    free_feed_forward_network(network);

    return result;
}