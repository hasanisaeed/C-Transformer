#ifndef EMBEDDING_H
#define EMBEDDING_H

typedef struct
{
    int unique_id;
    char *token;
} WordEncoding;

WordEncoding *create_encoding(char **words, int word_count);
void free_embedding_matrix(char **words, int row);

#endif // EMBEDDING_H
