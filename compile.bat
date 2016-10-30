
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
nvcc kernel.cu tiny_obj_loader.cc -o a.exe
del a.lib
del a.exp
pause
