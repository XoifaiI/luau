@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set NINJA="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
cd /d C:\Users\Jack\Documents\luau-fork
if not exist build mkdir build
%CMAKE% -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=%NINJA% -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLUAU_BUILD_CLI=ON
if errorlevel 1 exit /b 1
%CMAKE% --build build --target Luau.Repl.CLI Luau.Compile.CLI -j 8
