@echo off
pushd %~dp0\..\
call premake5.exe vs2019 --file=Workspace.lua
popd
pause
