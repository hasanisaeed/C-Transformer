
#ifndef PREDICT_WORD_H
#define PREDICT_WORD_H

#include "../decoder/decoder.h"
#include "../common/embedding.h"

char *predict_word(DecoderOutput *decoder_output, WordEncoding *words_encoding, int vocab_size, char **vocabulary);

#endif // PREDICT_WORD_H