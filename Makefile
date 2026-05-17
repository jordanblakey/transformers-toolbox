CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
TARGET = transformer_layer
SRC = transformer_layer.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	del $(TARGET).exe 2>NUL || rm -f $(TARGET).exe $(TARGET)
