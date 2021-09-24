CXX := nvcc
TARGET := convolution
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11

all: convolution oop

convolution: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) -lcudnn

oop: oop.cu ConvolutionLayer.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) oop.cu ConvolutionLayer.cu -o oop -lcudnn

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
