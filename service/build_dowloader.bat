@echo off
rem ---------------------------------------------------------------
rem ComfyUI Downloader Build Script
rem ---------------------------------------------------------------
rem This batch file builds a standalone executable from downloader.py
rem using PyInstaller. The resulting executable (model_info.exe) will
rem handle model information gathering without requiring a Python
rem installation on the target machine.
rem
rem Options:
rem -F               : Create a single-file executable
rem -n model_info.exe: Name the output executable as model_info.exe
rem --exclude-module : Exclude specific Python modules from the package
rem                    to reduce executable size and avoid dependencies
rem                    that aren't needed for the downloader functionality
rem ---------------------------------------------------------------

pyinstaller -F downloader.py -n model_info.exe ^
--exclude-module scipy ^
--exclude-module sqlite3 ^
--exclude-module tensorflow ^
--exclude-module _tkinter ^
--exclude-module nltk ^
--exclude-module torch ^
--exclude-module sklearn ^
--exclude-module numpy ^
--exclude-module PIL ^
--exclude-module jinja2 