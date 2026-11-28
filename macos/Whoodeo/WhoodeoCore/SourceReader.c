#include "SourceReader.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Frame identity is the count of pictures decoded since the previous index
// keyframe. Packed MPEG-4 puts the neighboring picture's timestamp on the
// frame, so the decoder PTS is not the index.

struct SourceFile {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVPacket *packet;
    AVFrame *frame;
    struct SwsContext *sws;
    int swsFormat;
    int swsWidth;
    int swsHeight;
    int swsBits;
    uint8_t *rgb;
    int rgbBytes;
    int64_t *pts;
    uint8_t *key;
    int count;
    int video;
    int width;
    int height;
    int bits;
    double fps;
    AVRational timeBase;
    int nextIndex;
    int positioned;
    int inputEOF;
    int audioCount;
    int fast;
    int64_t startPts;
    AVRational frameRate;
    int clipped;
    AVFrame *held;
    int heldReady;
    char error[512];
};

struct SourceAudio {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVPacket *packet;
    AVFrame *frame;
    struct SwrContext *swr;
    int stream;
    int channels;
    int rate;
    int64_t startSample;
    int64_t endSample;
    int64_t nextSample;
    int inputEOF;
    int heldPacket;
    int finished;
    int16_t *pending;
    int pendingCount;
    int pendingPos;
    int64_t pendingSample;
    char error[512];
};

static void silence_logs(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    av_log_set_level(AV_LOG_QUIET);
}

static void set_error(char *error, int length, const char *text) {
    if (error == NULL || length <= 0) return;
    snprintf(error, (size_t)length, "%s", text);
}

static void set_status(char *error, int length, int status, const char *what) {
    char detail[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, detail, sizeof detail);
    if (error == NULL || length <= 0) return;
    snprintf(error, (size_t)length, "%s: %s", what, detail);
}

static int grow_timeline(SourceFile *file, int count) {
    int64_t *pts = av_realloc(file->pts, (size_t)count * sizeof(int64_t));
    if (pts == NULL) return -1;
    file->pts = pts;
    uint8_t *key = av_realloc(file->key, (size_t)count);
    if (key == NULL) return -1;
    file->key = key;
    return 0;
}

// False when the index or packet points past the end of a truncated file.
static int data_in_file(const SourceFile *file, int64_t pos, int size) {
    if (file->fmt == NULL || file->fmt->pb == NULL) return 1;
    int64_t fileSize = avio_size(file->fmt->pb);
    if (fileSize <= 0 || pos < 0) return 1;
    if (pos >= fileSize) return 0;
    if (size > 0 && pos + (int64_t)size > fileSize) return 0;
    return 1;
}

static int timeline_from_index(SourceFile *file, AVStream *stream) {
    int entries = avformat_index_get_entries_count(stream);
    if (entries <= 0) return -1;
    if (grow_timeline(file, entries) != 0) return -1;
    int count = 0;
    int64_t previous = INT64_MIN;
    int backward = 0;
    int clipped = 0;
    for (int i = 0; i < entries; i++) {
        const AVIndexEntry *entry = avformat_index_get_entry(stream, i);
        if (entry == NULL) return -1;
        if (entry->flags & AVINDEX_DISCARD_FRAME) continue;
        // An MP4 moov can list samples whose bytes sit past a truncated mdat.
        if (!data_in_file(file, entry->pos, entry->size)) {
            clipped++;
            continue;
        }
        if (entry->timestamp < previous) backward++;
        previous = entry->timestamp;
        file->pts[count] = entry->timestamp;
        file->key[count] = (entry->flags & AVINDEX_KEYFRAME) ? 1 : 0;
        count++;
    }
    if (count == 0 || backward > 0) return -1;
    int64_t firstGap = count > 1 ? file->pts[1] - file->pts[0] : 1;
    if (firstGap <= 0) firstGap = 1;
    int big = 0;
    for (int i = 1; i < count; i++) {
        int64_t gap = file->pts[i] - file->pts[i - 1];
        if (gap > firstGap * 2 + firstGap / 2) big++;
    }
    // A keyframe-only index steps by a whole GOP. Read the packets instead.
    if (big * 2 > count) return -1;
    int64_t declared = stream->nb_frames;
    // A complete index on a truncated file is shorter than nb_frames / duration.
    // Keep it; falling back to a packet scan would only rediscover the same cut.
    if (!clipped && declared > 0 && count < declared) return -1;
    // Matroska often indexes only the keyframes it has seen so far, sometimes just
    // the first. When the index holds far fewer pictures than duration times rate,
    // read the packets instead.
    double seconds = stream->duration != AV_NOPTS_VALUE && stream->duration > 0
        ? (double)stream->duration * av_q2d(stream->time_base)
        : (file->fmt->duration > 0 ? (double)file->fmt->duration / AV_TIME_BASE : 0);
    double rate = av_q2d(av_guess_frame_rate(file->fmt, stream, NULL));
    if (!clipped && seconds > 0 && rate > 0 && count < 0.9 * seconds * rate) return -1;
    file->count = count;
    file->clipped = clipped > 0;
    return 0;
}

typedef struct {
    int64_t pts;
    int32_t order;
    uint8_t key;
} Stamp;

static int stamp_compare(const void *left, const void *right) {
    const Stamp *a = left;
    const Stamp *b = right;
    if (a->pts < b->pts) return -1;
    if (a->pts > b->pts) return 1;
    if (a->order < b->order) return -1;
    if (a->order > b->order) return 1;
    return 0;
}

static int timeline_from_packets(SourceFile *file, AVStream *stream) {
    (void)stream;
    int capacity = 4096;
    Stamp *stamps = av_malloc((size_t)capacity * sizeof(Stamp));
    AVPacket *packet = av_packet_alloc();
    if (stamps == NULL || packet == NULL) {
        av_free(stamps);
        av_packet_free(&packet);
        return -1;
    }
    int count = 0;
    while (av_read_frame(file->fmt, packet) >= 0) {
        if (packet->stream_index != file->video) {
            av_packet_unref(packet);
            continue;
        }
        if (count == capacity) {
            capacity *= 2;
            Stamp *grown = av_realloc(stamps, (size_t)capacity * sizeof(Stamp));
            if (grown == NULL) {
                av_free(stamps);
                av_packet_free(&packet);
                return -1;
            }
            stamps = grown;
        }
        if ((packet->flags & AV_PKT_FLAG_CORRUPT)
            || !data_in_file(file, packet->pos, packet->size)) {
            file->clipped = 1;
            av_packet_unref(packet);
            continue;
        }
        stamps[count].pts = packet->pts == AV_NOPTS_VALUE ? packet->dts : packet->pts;
        stamps[count].order = count;
        stamps[count].key = (packet->flags & AV_PKT_FLAG_KEY) ? 1 : 0;
        count++;
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    if (count == 0) {
        av_free(stamps);
        return -1;
    }
    qsort(stamps, (size_t)count, sizeof(Stamp), stamp_compare);
    if (grow_timeline(file, count) != 0) {
        av_free(stamps);
        return -1;
    }
    for (int i = 0; i < count; i++) {
        file->pts[i] = stamps[i].pts;
        file->key[i] = stamps[i].key;
    }
    av_free(stamps);
    file->count = count;
    return 0;
}

static int open_decoder(SourceFile *file, AVStream *stream, char *error, int errorLength) {
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (codec == NULL) {
        set_error(error, errorLength, "für dieses Bild gibt es keinen Decoder");
        return -1;
    }
    file->dec = avcodec_alloc_context3(codec);
    if (file->dec == NULL) {
        set_error(error, errorLength, "Decoder startet nicht");
        return -1;
    }
    int status = avcodec_parameters_to_context(file->dec, stream->codecpar);
    if (status < 0) {
        set_status(error, errorLength, status, "Decoder startet nicht");
        return -1;
    }
    file->dec->pkt_timebase = stream->time_base;
    // One thread. Frame threading would add pictures the index count does not expect.
    file->dec->thread_count = 1;
    status = avcodec_open2(file->dec, codec, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Decoder startet nicht");
        return -1;
    }
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(file->dec->pix_fmt);
    file->bits = desc != NULL && desc->comp[0].depth > 0 ? desc->comp[0].depth : 8;
    file->width = file->dec->width > 0 ? file->dec->width : stream->codecpar->width;
    file->height = file->dec->height > 0 ? file->dec->height : stream->codecpar->height;
    file->packet = av_packet_alloc();
    file->frame = av_frame_alloc();
    if (file->packet == NULL || file->frame == NULL) {
        set_error(error, errorLength, "Decoder startet nicht");
        return -1;
    }
    return 0;
}

static int arrive(SourceFile *file, int index);
static int clip_unreadable_tail(SourceFile *file);

SourceFile *source_open(const char *path, char *error, int errorLength) {
    silence_logs();
    SourceFile *file = av_mallocz(sizeof(SourceFile));
    if (file == NULL) {
        set_error(error, errorLength, "Film lässt sich nicht öffnen");
        return NULL;
    }
    file->video = -1;
    file->nextIndex = -1;
    int status = avformat_open_input(&file->fmt, path, NULL, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Film lässt sich nicht öffnen");
        source_close(file);
        return NULL;
    }
    status = avformat_find_stream_info(file->fmt, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Film lässt sich nicht öffnen");
        source_close(file);
        return NULL;
    }
    file->video = av_find_best_stream(file->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (file->video < 0) {
        set_error(error, errorLength, "kein Bild in der Datei");
        source_close(file);
        return NULL;
    }
    AVStream *stream = file->fmt->streams[file->video];
    file->timeBase = stream->time_base.num > 0 && stream->time_base.den > 0
        ? stream->time_base
        : (AVRational){1, 25};
    if (timeline_from_index(file, stream) != 0) {
        av_freep(&file->pts);
        av_freep(&file->key);
        file->count = 0;
        if (timeline_from_packets(file, stream) != 0) {
            set_error(error, errorLength, "keine Bilder in der Datei");
            source_close(file);
            return NULL;
        }
    }
    if (open_decoder(file, stream, error, errorLength) != 0) {
        source_close(file);
        return NULL;
    }
    if (file->clipped && clip_unreadable_tail(file) != 0) {
        set_error(error, errorLength, "keine Bilder in der Datei");
        source_close(file);
        return NULL;
    }
    if (file->count >= 2 && file->pts[1] > file->pts[0]) {
        double seconds = (double)(file->pts[1] - file->pts[0]) * av_q2d(file->timeBase);
        file->fps = seconds > 0 ? 1.0 / seconds : 25;
    } else {
        AVRational rate = av_guess_frame_rate(file->fmt, stream, NULL);
        file->fps = av_q2d(rate);
        if (!(file->fps > 0)) file->fps = 25;
    }
    for (unsigned i = 0; i < file->fmt->nb_streams; i++) {
        if (file->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) file->audioCount++;
    }
    return file;
}


SourceFile *source_open_fast(const char *path, char *error, int errorLength) {
    silence_logs();
    SourceFile *file = av_mallocz(sizeof(SourceFile));
    if (file == NULL) {
        set_error(error, errorLength, "Film lässt sich nicht öffnen");
        return NULL;
    }
    file->video = -1;
    file->nextIndex = -1;
    file->fast = 1;
    int status = avformat_open_input(&file->fmt, path, NULL, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Film lässt sich nicht öffnen");
        source_close(file);
        return NULL;
    }
    status = avformat_find_stream_info(file->fmt, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Film lässt sich nicht öffnen");
        source_close(file);
        return NULL;
    }
    file->video = av_find_best_stream(file->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (file->video < 0) {
        set_error(error, errorLength, "kein Bild in der Datei");
        source_close(file);
        return NULL;
    }
    AVStream *stream = file->fmt->streams[file->video];
    file->timeBase = stream->time_base.num > 0 && stream->time_base.den > 0
        ? stream->time_base
        : (AVRational){1, 25};
    AVRational rate = stream->avg_frame_rate;
    if (rate.num <= 0 || rate.den <= 0) rate = stream->r_frame_rate;
    if (rate.num <= 0 || rate.den <= 0) rate = av_guess_frame_rate(file->fmt, stream, NULL);
    if (rate.num <= 0 || rate.den <= 0) rate = (AVRational){25, 1};
    file->frameRate = rate;
    file->fps = av_q2d(rate);
    if (!(file->fps > 0)) {
        file->frameRate = (AVRational){25, 1};
        file->fps = 25;
    }
    file->startPts = stream->start_time != AV_NOPTS_VALUE ? stream->start_time : 0;
    if (stream->nb_frames > 0) {
        file->count = (int)stream->nb_frames;
    } else {
        double seconds = 0;
        if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
            seconds = (double)stream->duration * av_q2d(stream->time_base);
        } else if (file->fmt->duration > 0) {
            seconds = (double)file->fmt->duration / AV_TIME_BASE;
        }
        if (seconds > 0 && file->fps > 0) {
            int frames = (int)llround(seconds * file->fps);
            file->count = frames > 0 ? frames : 0;
        }
    }
    if (open_decoder(file, stream, error, errorLength) != 0) {
        source_close(file);
        return NULL;
    }
    for (unsigned i = 0; i < file->fmt->nb_streams; i++) {
        if (file->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) file->audioCount++;
    }
    return file;
}

void source_close(SourceFile *file) {
    if (file == NULL) return;
    sws_freeContext(file->sws);
    av_freep(&file->rgb);
    av_frame_free(&file->held);
    av_frame_free(&file->frame);
    av_packet_free(&file->packet);
    avcodec_free_context(&file->dec);
    avformat_close_input(&file->fmt);
    av_freep(&file->pts);
    av_freep(&file->key);
    av_free(file);
}

const char *source_error(const SourceFile *file) {
    return file != NULL ? file->error : "Film fehlt";
}

int source_width(const SourceFile *file) { return file->width; }
int source_height(const SourceFile *file) { return file->height; }
int source_count(const SourceFile *file) { return file->count; }
int source_bits(const SourceFile *file) { return file->bits; }
double source_fps(const SourceFile *file) { return file->fps; }

int source_clipped(const SourceFile *file) { return file != NULL ? file->clipped : 0; }

int source_declared(const SourceFile *file) {
    if (file == NULL || file->fmt == NULL || file->video < 0) return 0;
    int64_t declared = file->fmt->streams[file->video]->nb_frames;
    return declared > 0 && declared < INT32_MAX ? (int)declared : 0;
}

const char *source_codec(const SourceFile *file) {
    if (file == NULL || file->fmt == NULL || file->video < 0) return "";
    return avcodec_get_name(file->fmt->streams[file->video]->codecpar->codec_id);
}

static int64_t fast_pts(const SourceFile *file, int index) {
    AVRational rate = file->frameRate;
    if (rate.num <= 0 || rate.den <= 0) rate = (AVRational){25, 1};
    return file->startPts + av_rescale_q((int64_t)index, av_inv_q(rate), file->timeBase);
}

int source_time(const SourceFile *file, int index, int64_t *value, int32_t *timescale) {
    if (index < 0 || index >= file->count) return -1;
    int64_t pts = file->fast || file->pts == NULL ? fast_pts(file, index) : file->pts[index];
    *value = pts * (int64_t)file->timeBase.num;
    *timescale = file->timeBase.den > 0 ? file->timeBase.den : 1;
    return 0;
}

// Container duration in seconds. Matroska often leaves the stream duration unset.
static double container_seconds(const SourceFile *file) {
    if (file == NULL || file->fmt == NULL) return 0;
    if (file->video >= 0) {
        AVStream *stream = file->fmt->streams[file->video];
        if (stream != NULL && stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
            return (double)stream->duration * av_q2d(stream->time_base);
        }
    }
    if (file->fmt->duration > 0) {
        return (double)file->fmt->duration / (double)AV_TIME_BASE;
    }
    return 0;
}

// The MKV header of a 2/1 low file often keeps the master's frame rate
// (`-fps_mode passthrough`). When the catalog has an authoritative frame
// count, pick the rational (header, header/2, or a standard rate such as
// 25/1) whose implied count is closest to that. A short packet-step would
// drift on 29.97 files, because Matroska rounds those to 33/34 ms.
static void snap_fast_rate(SourceFile *file, int count) {
    if (file == NULL || !file->fast || count < 1) return;
    double seconds = container_seconds(file);
    if (!(seconds > 0)) return;
    AVRational header = file->frameRate;
    if (header.num <= 0 || header.den <= 0) header = (AVRational){25, 1};

    AVRational raw[] = {
        header,
        av_mul_q(header, (AVRational){1, 2}),
        av_mul_q(header, (AVRational){1, 3}),
        {24, 1}, {25, 1}, {30, 1}, {50, 1}, {60, 1},
        {24000, 1001}, {30000, 1001}, {60000, 1001}, {120000, 1001},
    };
    AVRational best = header;
    double bestErr = 1e300;
    for (size_t i = 0; i < sizeof raw / sizeof raw[0]; i++) {
        int num = raw[i].num, den = raw[i].den;
        if (num <= 0 || den <= 0) continue;
        av_reduce(&num, &den, num, den, INT_MAX);
        AVRational cand = {num, den};
        double fps = av_q2d(cand);
        if (!(fps > 0)) continue;
        double err = fabs(seconds * fps - (double)count);
        if (err < bestErr) {
            bestErr = err;
            best = cand;
        }
    }
    // More than 2 % off is not a plausible rate; keep the header.
    if (bestErr > 0.02 * (double)count) return;
    file->frameRate = best;
    file->fps = av_q2d(best);
}

int source_set_count(SourceFile *file, int count) {
    if (file == NULL || !file->fast || count < 1) return -1;
    file->count = count;
    snap_fast_rate(file, count);
    return 0;
}

static void drop_held(SourceFile *file) {
    if (file->held != NULL) av_frame_unref(file->held);
    file->heldReady = 0;
}

static int decode_one(SourceFile *file, AVFrame *frame) {
    if (file->heldReady && file->held != NULL) {
        av_frame_unref(frame);
        av_frame_move_ref(frame, file->held);
        file->heldReady = 0;
        return 0;
    }
    for (;;) {
        int got = avcodec_receive_frame(file->dec, frame);
        if (got == 0) return 0;
        if (got != AVERROR(EAGAIN)) return got;
        if (file->inputEOF) {
            int sent = avcodec_send_packet(file->dec, NULL);
            if (sent < 0 && sent != AVERROR_EOF) return sent;
            continue;
        }
        int read = av_read_frame(file->fmt, file->packet);
        if (read == AVERROR_EOF) {
            file->inputEOF = 1;
            continue;
        }
        if (read < 0) return read;
        if (file->packet->stream_index != file->video) {
            av_packet_unref(file->packet);
            continue;
        }
        int sent = avcodec_send_packet(file->dec, file->packet);
        av_packet_unref(file->packet);
        if (sent < 0) return sent;
    }
}

// True once a packet is at or past the index time of the wanted keyframe.
// MP4 with B-frames indexes decode times, so either stamp may be the one that matches.
static int reaches(const AVPacket *packet, int64_t target) {
    if (packet->dts == AV_NOPTS_VALUE && packet->pts == AV_NOPTS_VALUE) return 1;
    return (packet->dts != AV_NOPTS_VALUE && packet->dts >= target)
        || (packet->pts != AV_NOPTS_VALUE && packet->pts >= target);
}

static int position(SourceFile *file, int index) {
    int key = index;
    while (key > 0 && file->key[key] == 0) key--;
    int status = av_seek_frame(file->fmt, file->video, file->pts[key], AVSEEK_FLAG_BACKWARD);
    if (status < 0) return status;
    avcodec_flush_buffers(file->dec);
    file->inputEOF = 0;
    drop_held(file);
    // A backward seek can land on the keyframe before the wanted one, a whole GOP early.
    // Skip packets until the wanted keyframe, or every frame after it is off by that GOP.
    for (;;) {
        status = av_read_frame(file->fmt, file->packet);
        if (status < 0) return status;
        if (file->packet->stream_index == file->video
            && (file->packet->flags & AV_PKT_FLAG_KEY)
            && reaches(file->packet, file->pts[key])) break;
        av_packet_unref(file->packet);
    }
    status = avcodec_send_packet(file->dec, file->packet);
    av_packet_unref(file->packet);
    if (status < 0) return status;
    for (int i = 0; i < index - key; i++) {
        status = decode_one(file, file->frame);
        if (status < 0) return status;
        av_frame_unref(file->frame);
    }
    file->nextIndex = index;
    file->positioned = 1;
    return 0;
}

static int convert(SourceFile *file, const AVFrame *frame, float *rgb, int capacity, int *width, int *height) {
    int w = frame->width;
    int h = frame->height;
    *width = w;
    *height = h;
    if (w <= 0 || h <= 0) return -1;
    if (capacity < w * h * 3) {
        snprintf(file->error, sizeof file->error, "Bildpuffer ist zu klein");
        return -1;
    }
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((enum AVPixelFormat)frame->format);
    int bits = desc != NULL && desc->comp[0].depth > 8 ? 16 : 8;
    enum AVPixelFormat dest = bits > 8 ? AV_PIX_FMT_RGB48LE : AV_PIX_FMT_RGB24;
    int pixelBytes = bits > 8 ? 6 : 3;
    if (file->sws == NULL || file->swsFormat != frame->format || file->swsWidth != w || file->swsHeight != h || file->swsBits != bits) {
        sws_freeContext(file->sws);
        file->sws = sws_getContext(
            w, h, (enum AVPixelFormat)frame->format,
            w, h, dest,
            SWS_BILINEAR, NULL, NULL, NULL
        );
        file->swsFormat = frame->format;
        file->swsWidth = w;
        file->swsHeight = h;
        file->swsBits = bits;
        if (file->sws == NULL) {
            snprintf(file->error, sizeof file->error, "Farbwandlung startet nicht");
            return -1;
        }
    }
    int bytes = w * h * pixelBytes;
    if (file->rgbBytes < bytes) {
        uint8_t *grown = av_realloc(file->rgb, (size_t)bytes);
        if (grown == NULL) {
            snprintf(file->error, sizeof file->error, "Bildpuffer fehlt");
            return -1;
        }
        file->rgb = grown;
        file->rgbBytes = bytes;
    }
    uint8_t *planes[4] = { file->rgb, NULL, NULL, NULL };
    int stride[4] = { w * pixelBytes, 0, 0, 0 };
    sws_scale(file->sws, (const uint8_t *const *)frame->data, frame->linesize, 0, h, planes, stride);
    int plane = w * h;
    if (bits > 8) {
        const uint16_t *sample = (const uint16_t *)file->rgb;
        for (int i = 0; i < plane; i++) {
            rgb[i] = (float)sample[i * 3] / 65535.0f;
            rgb[plane + i] = (float)sample[i * 3 + 1] / 65535.0f;
            rgb[plane * 2 + i] = (float)sample[i * 3 + 2] / 65535.0f;
        }
    } else {
        const uint8_t *sample = file->rgb;
        for (int i = 0; i < plane; i++) {
            rgb[i] = (float)sample[i * 3] / 255.0f;
            rgb[plane + i] = (float)sample[i * 3 + 1] / 255.0f;
            rgb[plane * 2 + i] = (float)sample[i * 3 + 2] / 255.0f;
        }
    }
    return 0;
}

// Puts the decoder right before `index`. A short step forward decodes through instead of seeking,
// so a film that skips its repeated frames still reads in order.
static int64_t frame_pts(const AVFrame *frame) {
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) return frame->best_effort_timestamp;
    if (frame->pts != AV_NOPTS_VALUE) return frame->pts;
    return frame->pkt_dts;
}

// Seek backward to the keyframe before the target PTS, then decode forward
// and keep the picture whose PTS is nearest the target. 2/1 files can be
// slightly VFR around an otherwise regular grid (fblend of a 59.94 master),
// so the first PTS at or past the target can be the neighbour.
static int position_fast(SourceFile *file, int index) {
    int64_t target = fast_pts(file, index);
    int status = av_seek_frame(file->fmt, file->video, target, AVSEEK_FLAG_BACKWARD);
    if (status < 0) return status;
    avcodec_flush_buffers(file->dec);
    file->inputEOF = 0;
    drop_held(file);
    AVFrame *prev = av_frame_alloc();
    if (prev == NULL) return AVERROR(ENOMEM);
    int havePrev = 0;
    for (;;) {
        status = decode_one(file, file->frame);
        if (status < 0) {
            // Past the last picture: use it if we have one.
            if (havePrev && (status == AVERROR_EOF || status == AVERROR(EPIPE))) {
                av_frame_unref(file->frame);
                av_frame_move_ref(file->frame, prev);
                av_frame_free(&prev);
                file->nextIndex = index;
                file->positioned = 1;
                return 0;
            }
            av_frame_free(&prev);
            return status;
        }
        int64_t pts = frame_pts(file->frame);
        if (pts == AV_NOPTS_VALUE || pts >= target) {
            if (havePrev && pts != AV_NOPTS_VALUE) {
                int64_t prevPts = frame_pts(prev);
                if (prevPts != AV_NOPTS_VALUE) {
                    int64_t toCur = pts - target;
                    int64_t toPrev = target - prevPts;
                    if (toPrev <= toCur) {
                        // Keep the overshot picture so a following sequential
                        // read (t, t+1) does not skip a frame.
                        if (file->held == NULL) {
                            file->held = av_frame_alloc();
                            if (file->held == NULL) {
                                av_frame_unref(file->frame);
                                av_frame_move_ref(file->frame, prev);
                                av_frame_free(&prev);
                                file->nextIndex = index;
                                file->positioned = 1;
                                return 0;
                            }
                        }
                        av_frame_unref(file->held);
                        av_frame_move_ref(file->held, file->frame);
                        file->heldReady = 1;
                        av_frame_move_ref(file->frame, prev);
                    }
                }
            }
            av_frame_free(&prev);
            file->nextIndex = index;
            file->positioned = 1;
            return 0;
        }
        av_frame_unref(prev);
        av_frame_move_ref(prev, file->frame);
        havePrev = 1;
    }
}

static int arrive(SourceFile *file, int index) {
    if (index < 0 || index >= file->count) {
        snprintf(file->error, sizeof file->error, "Bild %d fehlt", index);
        return -1;
    }
    if (file->positioned && index > file->nextIndex && index - file->nextIndex <= 8) {
        while (file->nextIndex < index) {
            if (decode_one(file, file->frame) < 0) {
                file->positioned = 0;
                break;
            }
            av_frame_unref(file->frame);
            file->nextIndex++;
        }
    }
    if (!file->positioned || index != file->nextIndex) {
        int status = file->fast ? position_fast(file, index) : position(file, index);
        if (status < 0) {
            file->positioned = 0;
            snprintf(file->error, sizeof file->error, "Bild %d fehlt", index);
            return -1;
        }
        if (file->fast) return 0;
    }
    int status = decode_one(file, file->frame);
    if (status < 0) {
        file->positioned = 0;
        snprintf(file->error, sizeof file->error, "Bild %d fehlt", index);
        return -1;
    }
    return 0;
}

// Drop trailing pictures that the index kept but the decoder cannot produce,
// typically the last open GOP of a truncated file.
static int clip_unreadable_tail(SourceFile *file) {
    while (file->count > 0) {
        if (arrive(file, file->count - 1) == 0) {
            av_frame_unref(file->frame);
            file->positioned = 0;
            file->nextIndex = -1;
            return 0;
        }
        file->error[0] = '\0';
        file->count--;
        file->positioned = 0;
        file->nextIndex = -1;
    }
    return -1;
}

int source_frame(SourceFile *file, int index, float *rgb, int capacity, int *width, int *height) {
    if (arrive(file, index) != 0) return -1;
    int status = convert(file, file->frame, rgb, capacity, width, height);
    av_frame_unref(file->frame);
    if (status != 0) {
        file->positioned = 0;
        return -1;
    }
    file->nextIndex = index + 1;
    file->positioned = 1;
    return 0;
}

int source_luma(SourceFile *file, int index, uint8_t *luma, int capacity, int step, int *width, int *height) {
    if (step < 1) step = 1;
    if (arrive(file, index) != 0) return -1;
    AVFrame *frame = file->frame;
    int w = frame->width;
    int h = frame->height;
    int gw = (w + step - 1) / step;
    int gh = (h + step - 1) / step;
    *width = gw;
    *height = gh;
    int status = 0;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((enum AVPixelFormat)frame->format);
    if (w <= 0 || h <= 0 || capacity < gw * gh) {
        snprintf(file->error, sizeof file->error, "Bildpuffer ist zu klein");
        status = -1;
    } else if (desc != NULL && !(desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BE))
               && desc->comp[0].plane == 0 && desc->comp[0].step == (desc->comp[0].depth > 8 ? 2 : 1)) {
        // Planar or semi-planar YUV: the first plane is luma.
        int depth = desc->comp[0].depth;
        int shift = depth > 8 ? depth - 8 : 0;
        int stride = frame->linesize[0];
        for (int y = 0; y < gh; y++) {
            const uint8_t *row = frame->data[0] + (size_t)(y * step) * stride;
            uint8_t *out = luma + y * gw;
            if (depth > 8) {
                const uint16_t *wide = (const uint16_t *)row;
                for (int x = 0; x < gw; x++) out[x] = (uint8_t)(wide[x * step] >> shift);
            } else {
                for (int x = 0; x < gw; x++) out[x] = row[x * step];
            }
        }
    } else {
        // Anything else through the RGB conversion, luma from Rec. 709 weights.
        float *rgb = av_malloc(sizeof(float) * (size_t)w * h * 3);
        int cw = 0, ch = 0;
        if (rgb == NULL || convert(file, frame, rgb, w * h * 3, &cw, &ch) != 0) {
            snprintf(file->error, sizeof file->error, "Bild %d fehlt", index);
            status = -1;
        } else {
            int plane = w * h;
            for (int y = 0; y < gh; y++) {
                for (int x = 0; x < gw; x++) {
                    int i = y * step * w + x * step;
                    float v = 0.2126f * rgb[i] + 0.7152f * rgb[plane + i] + 0.0722f * rgb[2 * plane + i];
                    luma[y * gw + x] = (uint8_t)lrintf(fminf(fmaxf(v, 0), 1) * 255);
                }
            }
        }
        av_free(rgb);
    }
    av_frame_unref(frame);
    if (status != 0) {
        file->positioned = 0;
        return -1;
    }
    file->nextIndex = index + 1;
    file->positioned = 1;
    return 0;
}

static AVStream *audio_stream(AVFormatContext *fmt, int ordinal) {
    int seen = 0;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
        if (seen == ordinal) return fmt->streams[i];
        seen++;
    }
    return NULL;
}

int source_audio_count(const SourceFile *file) { return file->audioCount; }

int source_audio_info(const SourceFile *file, int ordinal, int *channels, double *sampleRate, double *bitRate) {
    AVStream *stream = audio_stream(file->fmt, ordinal);
    if (stream == NULL) return -1;
    int count = stream->codecpar->ch_layout.nb_channels;
    int rate = stream->codecpar->sample_rate;
    if (count <= 0 || rate <= 0) return -1;
    *channels = count;
    *sampleRate = rate;
    *bitRate = (double)stream->codecpar->bit_rate;
    return 0;
}

SourceAudio *source_audio_begin(const char *path, int ordinal, double start, double end, char *error, int errorLength) {
    silence_logs();
    SourceAudio *audio = av_mallocz(sizeof(SourceAudio));
    if (audio == NULL) {
        set_error(error, errorLength, "Ton lässt sich nicht öffnen");
        return NULL;
    }
    audio->stream = -1;
    int status = avformat_open_input(&audio->fmt, path, NULL, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Ton lässt sich nicht öffnen");
        source_audio_close(audio);
        return NULL;
    }
    status = avformat_find_stream_info(audio->fmt, NULL);
    if (status < 0) {
        set_status(error, errorLength, status, "Ton lässt sich nicht öffnen");
        source_audio_close(audio);
        return NULL;
    }
    AVStream *stream = audio_stream(audio->fmt, ordinal);
    if (stream == NULL) {
        set_error(error, errorLength, "Tonspur fehlt");
        source_audio_close(audio);
        return NULL;
    }
    audio->stream = -1;
    for (unsigned i = 0; i < audio->fmt->nb_streams; i++) {
        if (audio->fmt->streams[i] == stream) audio->stream = (int)i;
    }
    if (audio->stream < 0) {
        set_error(error, errorLength, "Tonspur fehlt");
        source_audio_close(audio);
        return NULL;
    }
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (codec == NULL) {
        set_error(error, errorLength, "für diesen Ton gibt es keinen Decoder");
        source_audio_close(audio);
        return NULL;
    }
    audio->dec = avcodec_alloc_context3(codec);
    if (audio->dec == NULL || avcodec_parameters_to_context(audio->dec, stream->codecpar) < 0) {
        set_error(error, errorLength, "Ton-Decoder startet nicht");
        source_audio_close(audio);
        return NULL;
    }
    audio->dec->pkt_timebase = stream->time_base;
    audio->dec->thread_count = 1;
    if (avcodec_open2(audio->dec, codec, NULL) < 0) {
        set_error(error, errorLength, "Ton-Decoder startet nicht");
        source_audio_close(audio);
        return NULL;
    }
    audio->channels = audio->dec->ch_layout.nb_channels;
    audio->rate = audio->dec->sample_rate;
    if (audio->channels <= 0 || audio->rate <= 0) {
        set_error(error, errorLength, "Tonformat fehlt");
        source_audio_close(audio);
        return NULL;
    }
    AVChannelLayout outLayout;
    av_channel_layout_default(&outLayout, audio->channels);
    status = swr_alloc_set_opts2(
        &audio->swr,
        &outLayout, AV_SAMPLE_FMT_S16, audio->rate,
        &audio->dec->ch_layout, audio->dec->sample_fmt, audio->dec->sample_rate,
        0, NULL
    );
    av_channel_layout_uninit(&outLayout);
    if (status < 0 || audio->swr == NULL || swr_init(audio->swr) < 0) {
        set_error(error, errorLength, "Tonwandlung startet nicht");
        source_audio_close(audio);
        return NULL;
    }
    audio->packet = av_packet_alloc();
    audio->frame = av_frame_alloc();
    if (audio->packet == NULL || audio->frame == NULL) {
        set_error(error, errorLength, "Ton-Decoder startet nicht");
        source_audio_close(audio);
        return NULL;
    }
    audio->startSample = (int64_t)llrint(start * audio->rate);
    audio->endSample = (int64_t)llrint(end * audio->rate);
    if (audio->endSample < audio->startSample) audio->endSample = audio->startSample;
    int64_t stamp = av_rescale_q((int64_t)llrint(start * AV_TIME_BASE), AV_TIME_BASE_Q, stream->time_base);
    status = av_seek_frame(audio->fmt, audio->stream, stamp, AVSEEK_FLAG_BACKWARD);
    if (status < 0) {
        set_status(error, errorLength, status, "Ton lässt sich nicht vorspulen");
        source_audio_close(audio);
        return NULL;
    }
    avcodec_flush_buffers(audio->dec);
    return audio;
}

void source_audio_close(SourceAudio *audio) {
    if (audio == NULL) return;
    swr_free(&audio->swr);
    av_freep(&audio->pending);
    av_frame_free(&audio->frame);
    av_packet_free(&audio->packet);
    avcodec_free_context(&audio->dec);
    avformat_close_input(&audio->fmt);
    av_free(audio);
}

int source_audio_channels(const SourceAudio *audio) { return audio->channels; }
double source_audio_rate(const SourceAudio *audio) { return audio->rate; }
const char *source_audio_error(const SourceAudio *audio) {
    return audio != NULL ? audio->error : "Ton fehlt";
}

// The first packets after a seek into packed MPEG audio are often not a
// complete frame. Skip those and keep going. A held packet stays until the
// decoder accepts it, because send can ask us to drain a frame first.
static int decode_audio(SourceAudio *audio) {
    int skipped = 0;
    int spins = 0;
    for (;;) {
        int got = avcodec_receive_frame(audio->dec, audio->frame);
        if (got == 0) return 0;
        if (got == AVERROR_EOF) return got;
        if (got != AVERROR(EAGAIN)) {
            av_frame_unref(audio->frame);
            if (got != AVERROR_INVALIDDATA || ++skipped > 256) return got;
        }
        if (audio->heldPacket) {
            int sent = avcodec_send_packet(audio->dec, audio->packet);
            if (sent == AVERROR(EAGAIN)) {
                if (++spins > 8) return sent;
                continue;
            }
            spins = 0;
            av_packet_unref(audio->packet);
            audio->heldPacket = 0;
            if (sent == AVERROR_INVALIDDATA) {
                if (++skipped > 256) return sent;
                continue;
            }
            if (sent < 0 && sent != AVERROR_EOF) return sent;
            skipped = 0;
            continue;
        }
        if (audio->inputEOF) {
            int sent = avcodec_send_packet(audio->dec, NULL);
            if (sent == AVERROR(EAGAIN)) continue;
            if (sent < 0 && sent != AVERROR_EOF) return sent;
            continue;
        }
        int read = av_read_frame(audio->fmt, audio->packet);
        if (read == AVERROR_EOF) {
            audio->inputEOF = 1;
            continue;
        }
        if (read < 0) return read;
        if (audio->packet->stream_index != audio->stream) {
            av_packet_unref(audio->packet);
            continue;
        }
        audio->heldPacket = 1;
    }
}

static int pull_audio(SourceAudio *audio) {
    if (audio->finished) return 1;
    int status = decode_audio(audio);
    if (status == AVERROR_EOF) {
        int delayed = swr_convert(audio->swr, NULL, 0, NULL, 0);
        (void)delayed;
        audio->finished = 1;
        av_frame_unref(audio->frame);
        return 1;
    }
    if (status < 0) {
        char detail[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(status, detail, sizeof detail);
        snprintf(audio->error, sizeof audio->error, "Ton bricht ab: %s", detail);
        return -1;
    }
    int64_t frameSample;
    if (audio->frame->pts != AV_NOPTS_VALUE) {
        AVRational base = audio->fmt->streams[audio->stream]->time_base;
        double seconds = audio->frame->pts * av_q2d(base);
        frameSample = (int64_t)llrint(seconds * audio->rate);
    } else {
        frameSample = audio->nextSample;
    }
    int outCap = swr_get_out_samples(audio->swr, audio->frame->nb_samples);
    if (outCap < audio->frame->nb_samples) outCap = audio->frame->nb_samples;
    int16_t *converted = av_malloc((size_t)outCap * (size_t)audio->channels * sizeof(int16_t));
    if (converted == NULL) {
        snprintf(audio->error, sizeof audio->error, "Tonpuffer fehlt");
        av_frame_unref(audio->frame);
        return -1;
    }
    uint8_t *planes[1] = { (uint8_t *)converted };
    int produced = swr_convert(
        audio->swr, planes, outCap,
        (const uint8_t **)audio->frame->extended_data, audio->frame->nb_samples
    );
    av_frame_unref(audio->frame);
    if (produced < 0) {
        av_free(converted);
        snprintf(audio->error, sizeof audio->error, "Tonwandlung bricht ab");
        return -1;
    }
    av_freep(&audio->pending);
    audio->pending = converted;
    audio->pendingCount = produced;
    audio->pendingPos = 0;
    audio->pendingSample = frameSample;
    audio->nextSample = frameSample + produced;
    while (audio->pendingPos < audio->pendingCount && audio->pendingSample < audio->startSample) {
        audio->pendingPos++;
        audio->pendingSample++;
    }
    if (audio->pendingSample >= audio->endSample) {
        audio->finished = 1;
        audio->pendingPos = audio->pendingCount;
        return 1;
    }
    return 0;
}

int source_audio_read(SourceAudio *audio, int16_t *dst, int maxFrames, int *framesOut, double *ptsOut) {
    int filled = 0;
    int64_t first = 0;
    while (filled < maxFrames) {
        if (audio->pendingPos >= audio->pendingCount) {
            int status = pull_audio(audio);
            if (status < 0) return -1;
            if (status > 0 || audio->pendingPos >= audio->pendingCount) break;
        }
        if (audio->pendingSample >= audio->endSample) {
            audio->finished = 1;
            break;
        }
        int available = audio->pendingCount - audio->pendingPos;
        int room = maxFrames - filled;
        int64_t untilEnd = audio->endSample - audio->pendingSample;
        int run = available < room ? available : room;
        if (untilEnd < run) run = (int)untilEnd;
        if (run <= 0) break;
        if (filled == 0) first = audio->pendingSample;
        memcpy(
            dst + (size_t)filled * (size_t)audio->channels,
            audio->pending + (size_t)audio->pendingPos * (size_t)audio->channels,
            (size_t)run * (size_t)audio->channels * sizeof(int16_t)
        );
        audio->pendingPos += run;
        audio->pendingSample += run;
        filled += run;
    }
    *framesOut = filled;
    if (filled == 0) return audio->error[0] != 0 ? -1 : 1;
    *ptsOut = (double)first / (double)audio->rate;
    return 0;
}
