@echo off
rem Run unittest with coverage, generate html output, and display it.
coverage run -m unittest %*
coverage html
open htmlcov\index.html