#include <math.h>
#include "positional_encoding.h"
#include "matrix_utils.h"

double **calculate_positional_encoding(int rows, int cols)
{
    double **positional_encoding_matrix = create_2d_matrix(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j % 2 == 0)
            {
                positional_encoding_matrix[i][j] = sin(i / pow(10000, j / (double)cols));
            }
            else
            {
                positional_encoding_matrix[i][j] = cos(i / pow(10000, (j - 1) / (double)cols));
            }
        }
    }

    return positional_encoding_matrix;
}
