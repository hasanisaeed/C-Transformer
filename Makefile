# Makefile

CC = gcc
CFLAGS = -Wall -I./src/utils -I./src/common

encoder: src/encoder/main_encoder.c src/encoder/encoder.c
	$(CC) $(CFLAGS) -o encoder src/encoder/main_encoder.c src/encoder/encoder.c src/common/*.c src/utils/*.c

decoder: src/decoder/main_decoder.c src/decoder/decoder.c
	$(CC) $(CFLAGS) -o decoder src/decoder/main_decoder.c src/decoder/decoder.c src/common/*.c src/utils/*.c