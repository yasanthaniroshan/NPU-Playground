CC = arm-rockchip830-linux-uclibcgnueabihf-gcc

TARGET = matmul.o
SRC = main.c

INCLUDES = -Iinclude
LIBS = -Llib -lrknnmrt

CFLAGS = 

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS) $(CFLAGS)

clean:
	rm -f $(TARGET)
