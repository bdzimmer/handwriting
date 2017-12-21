set OMP_NUM_THREADS=1

start "character classification training" cmd /c python -u -m handwriting.run_charclassml train ^> char_class_train_log.txt

start "character position training" cmd /c python -u -m handwriting.run_charposml train ^> char_pos_train_log.txt