# 
# Filename    : Makefile
# Author      : Yumi IWashita
# 

#CPP       = g++
#X_CFLAGS   += -Wall -O3 -DTC_LITTLE_ENDIAN 
#X_LIBS =  -L/usr/local/lib
CXXFLAGS := -g $(shell pkg-config --cflags opencv) -I/usr/include
#CXXFLAGS := $(shell pkg-config --cflags opencv) -I/usr/include
LDLIBS := $(shell pkg-config --libs opencv) /usr/lib/libann.so

CXXFLAGS += -MMD -std=gnu++11

objects = \
  main.o \
  classification_ibo.o \
  ibo.o \
  ibo/bco.o \
  ibo/buo.o \
  ibo/bfo.o \
  ibo/lbp.o \
  ibo/svm.o \
  ibo/libsvm.o \
  ibo/ann.o \
  ibo/pca.o \
  ibo/svm_linear.o \
  ibo/linearsvm/linear.o \
  ibo/linearsvm/tron.o \
  ibo/linearsvm/blas/daxpy.o \
  ibo/linearsvm/blas/ddot.o \
  ibo/linearsvm/blas/dnrm2.o \
  ibo/linearsvm/blas/dscal.o

sources = \
  main.cpp \
  classification_ibo.cpp \
  ibo.cpp \
  ibo/bco.cpp \
  ibo/buo.cpp \
  ibo/bfo.cpp \
  ibo/lbp.cpp \
  ibo/svm.cpp \
  ibo/libsvm.cpp \
  ibo/ann.cpp \
  ibo/pca.cpp \
  ibo/svm_linear.cpp \
  ibo/linearsvm/linear.cpp \
  ibo/linearsvm/tron.cpp \
  ibo/linearsvm/blas/dnrm2.c \
  ibo/linearsvm/blas/daxpy.c \
  ibo/linearsvm/blas/ddot.c \
  ibo/linearsvm/blas/dscal.c

headers = \
  main.h \
  classification_ibo.h \
  ibo.h \
  ibo/bco.h \
  ibo/buo.h \
  ibo/bfo.h \
  ibo/lbp.h \
  ibo/bop.h \
  ibo/svm.h \
  ibo/libsvm.h \
  ibo/ann.h \
  ibo/pca.h \
  ibo/libsvm.h \
  ibo/svm_linear.h \
  ibo/linearsvm/linear.h \
  ibo/linearsvm/tron.h \
  ibo/linearsvm/blas/blas.h \
  ibo/linearsvm/blas/blasp.h

program = \
  main

# Build Actions

main: $(objects)
	g++ $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	/bin/rm -f *.o *.a *~


# -----------------------------------------------------------------------------
# End
# -----------------------------------------------------------------------------

