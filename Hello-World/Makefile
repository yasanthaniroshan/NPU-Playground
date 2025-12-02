CC = arm-rockchip830-linux-uclibcgnueabihf-gcc
CFLAGS = -Wall -Wextra
TARGET = main
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
