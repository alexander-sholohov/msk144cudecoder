
The main project msk144cudecoder builds using MS Visual Studio 2019/2022 community edition. 
As in the Linux case:
```
mkdir _build
cd _build
cmake --build .
```

However, gfortran based project bpdecode.dll has to be built in MinGW/MSYS2 environment.

How to build bpdecode.dll for Windows using MSYS2 environment:

Install MSYS2 following the instruction https://www.msys2.org/
Install extra package into MSYS2: 
pacman -S cmake
pacman -S mingw-w64-x86_64-boost


Change current dir to project root.
```
mkdir _build_bpdecode
cd _build_bpdecode
cmake ../win32bpdecode
cmake --build .
```

Copy bpdecode.dll to the place where msk144cudecoder.exe will be running.

