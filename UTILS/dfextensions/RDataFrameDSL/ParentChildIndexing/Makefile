# Makefile for index_helpers shared library
#
# Usage:
#   make          - build libIndexHelpers.so
#   make test     - run basic tests
#   make clean    - remove built files
#
# Requirements:
#   - ROOT with root-config in PATH

CXX = g++
ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)

# Compiler flags
CXXFLAGS = -fPIC -O2 -Wall -Wextra $(ROOTCFLAGS)
LDFLAGS = -shared $(ROOTLIBS)

# Target
TARGET = libIndexHelpers.so
SOURCES = index_helpers.cxx
HEADERS = index_helpers.h

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES)
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Running basic test..."
	root -l -q -b 'test_index_helpers.C'
