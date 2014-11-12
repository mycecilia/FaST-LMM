@REM Installs Python on local command shell on GCD Gateway machine
@REM %1 = computer with user's Python configuration files (e.g. .continuum)

if "%1"=="" goto MISSING
set path=c:\GCD\esciencepy;c:\GCD\esciencepy\scripts;%path%
mkdir %userprofile%\.continuum
xcopy /d /e /s /c /h \\%1\c$\users\%username%\.continuum %userprofile%\.continuum
mkdir %userprofile%\.matplotlib
mkdir %userprofile%\.matplotlib\tex.cache
set MPLCONFIGDIR=%userprofile%\.matplotlib
mkdir %userprofile%\.ipython
set IPYTHONDIR=%userprofile%\.ipython
goto DONE

:MISSING
@echo ========ERROR==========
@echo This batch file requires the name of a computer machine on which you've installed your Anaconda/MKL license
@echo =======================
:DONE
