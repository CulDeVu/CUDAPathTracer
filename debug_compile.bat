
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
nvcc -G -g -lineinfo -arch=sm_20 kernel.cu tiny_obj_loader.obj -o a.exe
del a.lib
del a.exp
pause
