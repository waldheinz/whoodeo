#pragma once

#include <stdint.h>

// Source pictures and audio via the local FFmpeg libraries.
// The Homebrew build this links is GPL. A later build can swap in an LGPL FFmpeg.

typedef struct SourceFile SourceFile;
typedef struct SourceAudio SourceAudio;

SourceFile *source_open(const char *path, char *error, int errorLength);
void source_close(SourceFile *file);
const char *source_error(const SourceFile *file);

int source_width(const SourceFile *file);
int source_height(const SourceFile *file);
int source_count(const SourceFile *file);
int source_bits(const SourceFile *file);
double source_fps(const SourceFile *file);
int source_time(const SourceFile *file, int index, int64_t *value, int32_t *timescale);

// Writes planar float RGB, R then G then B, each sample in 0..1.
// 8-bit sources use the 0..255 steps. Wider sources use the 0..65535 steps.
int source_frame(SourceFile *file, int index, float *rgb, int capacity, int *width, int *height);

int source_audio_count(const SourceFile *file);
int source_audio_info(const SourceFile *file, int ordinal, int *channels, double *sampleRate, double *bitRate);

// Own demuxer, so an export can read audio while pictures are decoded.
// start and end are seconds. Samples at or after end are not returned.
SourceAudio *source_audio_begin(const char *path, int ordinal, double start, double end, char *error, int errorLength);
void source_audio_close(SourceAudio *audio);
int source_audio_channels(const SourceAudio *audio);
double source_audio_rate(const SourceAudio *audio);
// Interleaved signed 16-bit. Returns 0 with a chunk, 1 at the end, -1 on error.
int source_audio_read(SourceAudio *audio, int16_t *dst, int maxFrames, int *framesOut, double *ptsOut);
const char *source_audio_error(const SourceAudio *audio);
