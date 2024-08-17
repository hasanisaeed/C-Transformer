#include <stdio.h>
#include <stdlib.h>
#include "memory_utils.h"

char **allocate_memory(int capacity)
{
    char **words = (char **)malloc(capacity * sizeof(char *));
    if (words == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return words;
}

char **reallocate_memory(char **words, int new_capacity)
{
    words = (char **)realloc(words, new_capacity * sizeof(char *));
    if (words == NULL)
    {
        fprintf(stderr, "Memory reallocation failed\n");
        exit(EXIT_FAILURE);
    }
    return words;
}
