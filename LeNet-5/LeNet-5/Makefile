CXX			:= g++
CXXFLAGS	:= -Wall -Werror -Wextra -pedantic-errors
LDFLAGS		:= -lm

all: lenet

lenet: main.o lenet.o
	$(CXX) $(CFLAGS) -o lenet main.o lenet.o $(LDFLAGS)

%.o: %.c
	$(CXX) $(CFLAGS) -c $< -o $@


