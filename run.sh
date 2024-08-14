#!/usr/bin/env bash
make clean build

echo ""
echo "Run with default arguments"
echo ""

make run

echo ""
echo "Run with non-default arguments"
echo ""

make run ARGS="-threadsPerBlock=128 -notp=1024 -nopp 30 -thickness 0.5 -intensity=7.5 tau=10 wl0=900 dt=0.3 output=test"