OpenCL bindings for Haskell
===========================

These are OpenCL 1.2 bindings for Haskell. OpenCL resources are garbage
collected and OpenCL errors are thrown as exceptions.

GPGPU computing is some nifty stuff. Work in progress.

GHC 8.0 Trouble
---------------

Starting from GHC 8.0 the programs compiled with this Haskell compiler allocate
1 terabyte of unreserved memory at start up. While normally this is okay, on
Linux with nVidia's OpenCL library you may not be able to enumerate nVidia's
OpenCL platform.

Workaround is to use GHC 7.10 (or earlier) or recompile GHC 8.0 with
`--disable-large-address-space` to `./configure`.

