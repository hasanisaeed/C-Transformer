#include <math.h>
#include "add_norm.h"
#include "matrix_utils.h"
#include "memory_utils.h"

double **normalize_matrix(double **matrix, int rows, int cols)
{
    double **norm_matrix = create_2d_matrix(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        double mean = 0.0;
        for (int j = 0; j < cols; j++)
        {
            mean += matrix[i][j];
        }
        mean /= cols;

        double variance = 0.0;
        for (int j = 0; j < cols; j++)
        {
            variance += pow(matrix[i][j] - mean, 2);
        }
        variance /= cols;

        for (int j = 0; j < cols; j++)
        {
            norm_matrix[i][j] = (matrix[i][j] - mean) / sqrt(variance + 1e-8);
        }
    }

    return norm_matrix;
}

double **add_and_normalize(double **matrix1, double **matrix2, int rows, int cols)
{
    double **added = add_matrices(matrix1, matrix2, rows, cols);
    double **normalized = normalize_matrix(added, rows, cols);
    
    free_matrix(added, rows);
    return normalized;
}