#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

char **allocate_memory(int capacity);
char **reallocate_memory(char **words, int new_capacity);

#endif // MEMORY_UTILS_H
