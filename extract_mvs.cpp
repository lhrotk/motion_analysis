extern "C"{
#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libavcodec/mpegutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
#include <libavutil/avconfig.h>
}

#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
#define USE_REF_MEM 1
//#define MEASURE_TIME 1
#define ENABLE_ANALYSIS 1
//#define CALCULATE_ROI_AREA 1
//#define PRINT_MSG 1
using namespace cv;

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL;
static AVStream *video_stream = NULL;
static const char *src_filename = NULL;
static const char *dst_filename = NULL;
static AVCodecContext *c= NULL;
static FILE* f = NULL;
static AVPacket *pktEnc = NULL;

static int video_stream_idx = -1;
static AVFrame *frame = NULL;
static int video_frame_count = 0;
static int** prob;
static int** prob_last;
static int** rawRoi;
static int probH;
static int probW;
Mat roi;

typedef struct Memo{
	int skipCnt;
	int longestRef;
}Mem;


static Mem** mem;

void avframeToMat(const AVFrame * frame, Mat& image)
{
    int width = frame->width;
    int height = frame->height;

    // Allocate the opencv mat and store its stride in a 1-element array
    if (image.rows != height || image.cols != width || image.type() != CV_8UC3) image = Mat(height, width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = image.step1();

    // Convert the colour format and write directly to the opencv matrix
    SwsContext* conversion = sws_getContext(width, height, (AVPixelFormat) frame->format, width, height, AV_PIX_FMT_BGR24 , SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
    sws_freeContext(conversion);
}

static void probSet(int32_t *mb_type, int mv_cnt, AVMotionVector *mvs, bool is_IFrame){
	if (is_IFrame) {
		for (int i = 0; i < probH; i++) {
			for (int j = 0; j < probW; j++) {
				prob[i][j] = 0;
				prob_last[i][j] = 0;
				mem[i][j].longestRef = 0;
				mem[i][j].skipCnt = 0;
			}
		}
		return;
	}
	for (int i = 0; i < probH; i++) {
		for (int j = 0; j < probW; j++) {
			//if (IS_INTRA(frame->mb_type[i * (probW+1) + j])&&(i==0||i==probH-1||j==0||j==probW-1)) {
			if (IS_INTRA(frame->mb_type[i * (probW+1) + j])) {
				prob[i][j] = 100;
				mem[i][j].longestRef = 0;
				mem[i][j].skipCnt = 0;
			}
			else if(IS_SKIP(frame->mb_type[i * (probW+1)+ j]))
			{
				prob[i][j] = (int)(prob_last[i][j]*0.9);
#ifdef USE_REF_MEM
				mem[i][j].skipCnt++;
				if(mem[i][j].skipCnt>=mem[i][j].longestRef+3){
					prob[i][j] = 0;
					mem[i][j].longestRef = 0;
					mem[i][j].skipCnt = 0;
				}
#endif
			}
		}
	}
	for (int i = 0; i < mv_cnt; i++) {
		AVMotionVector* mv = &mvs[i];
		if(mv!=NULL&&!IS_SKIP(frame->mb_type[mv->dst_y/16*(probW+1)+mv->dst_x/16])&&!IS_INTRA(frame->mb_type[mv->dst_y/16*(probW+1)+mv->dst_x/16])){
			if(mv->dst_y/16<probH&&mv->src_y/16<probH&&mv->src_x/16<probW&&mv->dst_x/16<probW){
				int y = mv->src_y/16;
				int x = mv->src_x/16;
				int i = mv->dst_y/16;
				int j = mv->dst_x/16;
				int value = prob_last[y][x];
				/*if (y + 1 < probH)
					value = prob_last[y+1][x] > value ? prob_last[y+1][x] : value;
				if (y - 1 >= 0)
					value = prob_last[y-1][x] > value ? prob_last[y-1][x] : value;
				if (x + 1 < probW)
					value = prob_last[y][x+1] > value ? prob_last[y][x+1] : value;
				if (x - 1 >= 0)
					value = prob_last[y][x-1] > value ? prob_last[y][x-1] : value;*/
				if(i==y&&j==x){
					prob[i][j] = 0.9*value;
				}
				else{
					mem[i][j].longestRef++;
					mem[i][j].skipCnt=0;
					prob[i][j] = 1.1*value;
				}
			}
		}
	}
}

static void printProb() {
	int totalProb = 0;
	int nonProbCnt = 0;
	for (int i = 0; i < probH; i++) {
			for (int j = 0; j < probW; j++) {
				if(prob[i][j]>0){
					totalProb += prob[i][j];
					nonProbCnt++;
				}
			}
		}
	for (int i = 0; i < probH; i++) {
		for (int j = 0; j < probW; j++) {
			if(prob[i][j]>0.6*totalProb/nonProbCnt){
				rawRoi[i][j] = 1;
				if(!(i==0||j==0||i==probH-1||j==probW-1))
				roi.at<uchar>(i,j) = 255;
			}
			else{
				rawRoi[i][j] = 0;
			}
		}
	}
}

static void updateProb() {
	int** temp = prob_last;
	prob_last = prob;
	prob = temp;
	for (int i = 0; i < probH; i++) {
		for (int j = 0; j < probW; j++) {
			prob[i][j] = 0;
		}
	}
}


static void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;

    /* send the frame to the encoder */
    if (frame)
#ifdef PRINT_MSG
        printf("Send frame %3"PRId64"\n", frame->pts);
#endif

    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
#ifdef PRINT_MSG
        printf("Write packet %3"PRId64" (size=%5d)\n", pkt->pts, pkt->size);
#endif
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}

static int decode_packet(const AVPacket *pkt)
{
    int ret = avcodec_send_packet(video_dec_ctx, pkt);
    if (ret < 0) {
        //fprintf(stderr, "Error while sending a packet to the decoder: %s\n", av_err2str(ret));
        return ret;
    }

    while (ret >= 0)  {

        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            //fprintf(stderr, "Error while receiving a frame from the decoder: %s\n", av_err2str(ret));
            return ret;
        }
        frame->pts =  (1.0 / 30) * 90 * frame->display_picture_number;

        /*if(frame->qscale_table!=NULL)
        	fprintf(stderr, "mb001: %d\n", (int)frame->qscale_table[0]);
        if(frame->mb_type!=NULL){
        	if(IS_INTRA(frame->mb_type[0]))
                	fprintf(stderr, "mb %d: I\n", frame->coded_picture_number);
        }*/

#ifdef ENABLE_ANALYSIS
#ifdef PRINT_MSG
        fprintf(stdout, "=============frame: %d====================\n", frame->coded_picture_number);
#endif
        if(frame->pict_type==AV_PICTURE_TYPE_I){
        	probSet(NULL, 0, NULL, true);
        	return 0;
        }
#endif

        if (ret >= 0) {
#ifdef ENABLE_ANALYSIS
            int i;
            AVFrameSideData *sd;

            video_frame_count++;
            sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
            if (sd) {
                const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                /*for (i = 0; i < sd->size / sizeof(*mvs); i++) {
                    const AVMotionVector *mv = &mvs[i];
                    if(video_frame_count==-1){
                    	fprintf(stderr, "%d,%2d,%2d,%2d,%4d,%4d,%4d,%4d,0x%"PRIx64"\n",
                        video_frame_count, mv->source,
                        mv->w, mv->h, mv->src_x, mv->src_y,
                        mv->dst_x, mv->dst_y, mv->flags);
                    	if(mv->w==8){
                    		messageCnt++;
                    	}else{
                    		messageCnt +=2;
                    	}
                    	if(mv->h==8){
                    	    messageCnt++;
                    	}else{
                    	    messageCnt +=2;
                    	}
                    }else if(video_frame_count==3&&messageCnt!=0){
                    	fprintf(stderr, "cnt: %d\n", messageCnt);
                    	messageCnt = 0;
                    }
                }*/
            	probSet(frame->mb_type, sd->size / sizeof(*mvs), (AVMotionVector *)mvs, frame->pict_type==AV_PICTURE_TYPE_I);
            	printProb();
            	updateProb();

            }
            Mat image;
            avframeToMat(frame, image);
            for (int y = 0; y < c->height; y++) {
				for (int x = 0; x < c->width; x++) {
					if ((rawRoi[y / 16][x / 16] == 1)&&!(y/16==0||y/16==probH-1||x/16==0||x/16==probW-1)){
						//image.at<Vec3b>(Point(x,y)).val[2]=255;
					}
				}
			}
            dilate(roi, roi, Mat());
            for(int j=0; j<probW; j++){
            	roi.at<uchar>(0,j) = 0;
            	roi.at<uchar>(probH-1,j) = 0;
            }
			for (int i = 0; i < probH; i++) {
				roi.at<uchar>(i, probW - 1) = 0;
				roi.at<uchar>(i, 0) = 0;
			}
            imshow("roi", roi);
            Mat labels, img_color, stats,centroids;
            int nncomp = connectedComponentsWithStats(roi,labels,stats, centroids, 8);
#ifdef CALCULATE_ROI_AREA
            int roiTotalArea = 0;
#endif
            for(int i=0; i<nncomp; i++){
#ifdef PRINT_MSG
            	printf("area: %d\n", stats.at<int>(i, CC_STAT_AREA));
#endif
            	int x = 16*stats.at<int>(i, CC_STAT_LEFT);
            	int y = 16*stats.at<int>(i, CC_STAT_TOP);
            	int w = 16*stats.at<int>(i, CC_STAT_WIDTH);
            	int h = 16*stats.at<int>(i, CC_STAT_HEIGHT);
            	if(stats.at<int>(i, CC_STAT_AREA)>probH*probW/180&&stats.at<int>(i, CC_STAT_AREA)<probH*probW/2)
#ifdef CALCULATE_ROI_AREA
            	roiTotalArea += stats.at<int>(i, CC_STAT_AREA);
#endif
            	rectangle(image, Point(x,y),Point(x+w,y+h),Scalar(0,0,255),4);
            }
#ifdef PRINT_MSG
            printf("find connect region: %d\n", nncomp);
#endif
#ifdef CALCULATE_ROI_AREA
            fprintf(stderr, "Ratio: %lf\n", ((double)roiTotalArea)/(probH*probW));
#endif
        	roi.setTo(Scalar(0));
            imshow("test img", image);
#ifndef MEASURE_TIME
            waitKey(0);
#endif
#endif
            encode(c, frame, pktEnc, f);
            av_frame_unref(frame);
        }

    }

    return 0;
}

static int open_codec_context(AVFormatContext *fmt_ctx, enum AVMediaType type)
{
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = NULL;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename);
        return ret;
    } else {
        int stream_idx = ret;
        st = fmt_ctx->streams[stream_idx];

        dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx) {
            fprintf(stderr, "Failed to allocate codec\n");
            return AVERROR(EINVAL);
        }

        ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy codec parameters to codec context\n");
            return ret;
        }

        /* Init the video decoder */
        av_dict_set(&opts, "flags2", "+export_mvs", 0);
        if ((ret = avcodec_open2(dec_ctx, dec, &opts)) < 0) {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
            return ret;
        }

        video_stream_idx = stream_idx;
        video_stream = fmt_ctx->streams[video_stream_idx];
        video_dec_ctx = dec_ctx;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int ret = 0;
    AVPacket pkt = { 0 };
    const AVCodec *codec;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_video> <transcoded_video>\n", argv[0]);
        exit(1);
    }
    src_filename = argv[1];
    dst_filename = argv[2];

    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
        fprintf(stderr, "Could not open source file %s\n", src_filename);
        exit(1);
    }

    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO);

    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream) {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        ret = 1;
        goto end;
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        goto end;
    }

    /* find the libx264 encoder */
    codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) {
    	fprintf(stderr, "Codec libx264 not found\n");
    	goto end;
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        goto end;
    }

    /* put sample parameters */
	c->qmax = 30;
	c->qmin = 30;
	/* resolution must be a multiple of two */
	c->width = video_dec_ctx->width;
	c->height = video_dec_ctx->height;
	/* frames per second */
	c->time_base = (AVRational){1, 30};
	c->framerate = (AVRational){30, 1};

	/* emit one intra frame every 90 frames
	 * check frame pict_type before passing frame
	 * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
	 * then gop_size is ignored and the output of encoder
	 * will always be I frame irrespective to gop_size
	 */
	c->gop_size = 90;
	c->max_b_frames = 1;
	c->pix_fmt = AV_PIX_FMT_YUV420P;

	/* initialize the probability matrix*/
	probW = (int)ceil(c->width/16.0);
	probH = (int)ceil(c->height/16.0);
	prob = (int**)malloc(sizeof(int*)*probH);
	prob_last = (int**)malloc(sizeof(int*)*probH);
	mem = (Mem**)malloc(sizeof(Mem*)*probH);
	rawRoi = (int**)malloc(sizeof(int*)*probH);
	for(int i=0; i<probH; i++){
		prob[i] = (int*)malloc(sizeof(int)*probW);
		mem[i] = (Mem*)malloc(sizeof(Mem)*probW);
		prob_last[i] = (int*)malloc(sizeof(int)*probW);
		rawRoi[i] = (int*)malloc(sizeof(int)*probW);
	}


	namedWindow( "test img", 0 );
	roi = Mat(probH, probW, CV_8UC1);
	roi.setTo(Scalar(0));
	roi.at<uchar>(0,0) = 255;
	roi.setTo(Scalar(0));
	imshow("test", roi);
#ifndef MEASURE_TIME
	waitKey(100);
#endif

	if (codec->id == AV_CODEC_ID_H264)
		av_opt_set(c->priv_data, "preset", "veryfast", 0);

	/* open it */
	ret = avcodec_open2(c, codec, NULL);
	if (ret < 0) {
		//fprintf(stderr, "Could not open codec: %s\n", av_err2str(ret));
		exit(1);
	}

    /* open output file */
    f = fopen(dst_filename, "wb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", dst_filename);
        goto end;
    }

    pktEnc = av_packet_alloc();
    if (!pktEnc)
            goto end;

    printf("framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags\n");

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index == video_stream_idx){
            ret = decode_packet(&pkt);
        }
        av_packet_unref(&pkt);
        av_frame_unref(frame);
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_packet(NULL);
    encode(c, NULL, &pkt, f);

    fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);

end:
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    return ret < 0;
}
