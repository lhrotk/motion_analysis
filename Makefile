# use pkg-config for getting CFLAGS and LDLIBS
FFMPEG_LIBS=    libavdevice                        \
                libavformat                        \
                libavfilter                        \
                libavcodec                         \
                libswresample                      \
                libswscale                         \
                libavutil                          \
                opencv									

CFLAGS += -Wall -g 
CFLAGS := $(shell pkg-config --cflags $(FFMPEG_LIBS)) $(CFLAGS)
LDLIBS := -lstdc++ -std=c++11 $(shell pkg-config --libs $(FFMPEG_LIBS)) $(LDLIBS)

EXAMPLES +=       extract_mvs


OBJS=$(addsuffix .o,$(EXAMPLES))

.phony: all clean-test clean

all: $(OBJS) $(EXAMPLES)

clean-test:
	$(RM) test*.pgm test.h264 test.mp2 test.sw test.mpg

clean: clean-test
	$(RM) $(EXAMPLES) $(OBJS)
