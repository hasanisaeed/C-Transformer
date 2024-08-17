#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

#define LAYERS 1 // Number of [Linear Layer -> ReLU] layers

typedef struct
{
    int layer_sizes[LAYERS + 1];
    double ***weights;
    double ***biases;
} FeedForwardNetwork;

double **feed_forward(double **input, int rows, int cols, int *out_rows, int *out_cols);

FeedForwardNetwork *create_feed_forward_network(int layer_sizes[LAYERS + 1]);
double **add_bias(double **matrix, double **bias, int rows, int cols);
double **relu(double **matrix, int rows, int cols);

void free_feed_forward_network(FeedForwardNetwork *network);

#endif // FEED_FORWARD_NETWORK_H