REM takes one argument, the name of the log file (e.g. regression.log.txt), and redirects stdout and stderr there
python test.py > %1 2>&1