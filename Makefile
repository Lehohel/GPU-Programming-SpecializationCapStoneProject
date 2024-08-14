# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)
FILENAME = KerrEffect

all: clean build run

build:
	$(CXX) ./src/$(FILENAME).cu -o ./bin/$(FILENAME).exe -ccbin g++ -I./Common -lcudart -lcuda --std c++17

run:
	./bin/$(FILENAME).exe $(ARGS)

clean:
	rm -f ./bin/$(FILENAME).exe