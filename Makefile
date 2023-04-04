CC=gcc
CXX=g++

INC=-I./include \
	-I/usr/include \
	-I/usr/include/opencv2 \
	-I/usr/local/include/opencv4 \

LIB_DIR=/usr/lib \
		-L/usr/local/cuda/lib64 \
		
	
DEFINES =-D__EXPORTED_HEADERS__ 

CFLAGS =$(DEFINES) $(INC) -O0 -g -Wall -pthread -lm -lpthread -lstdc++
#-DUSE_CUDNN -DUSE_NCCL -MMD -MP
CXXFLAGS =$(DEFINES) $(INC) -std=c++11 -O0 -g -Wall -pthread -ggdb -fopenmp -lm -lpthread

LFLAGS =-Wl,-rpath-link=$(LIB_DIR) -L$(LIB_DIR) 
LFLAGS +=`pkg-config opencv --cflags --libs`  
LFLAGS +=`pkg-config opencv4 --cflags --libs`

OBJS = KalmanTracker.o Hungarian.o main.o 
BUILD = build
OBJDIR = obj
TARGET = Sort


all : $(TARGET)
	cd build
	sync
	
install : $(TARGET)

clean :
	rm -rf $(BUILD)/* $(OBJDIR)/*
	rm -rf *.o
	
$(TARGET) : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(CXXFLAGS) $(LFLAGS)
	mv $(OBJS) $(OBJDIR)
	mv $(TARGET) $(BUILD)
	


