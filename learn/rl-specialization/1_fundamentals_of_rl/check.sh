#!/bin/zsh
python dev_test.py > check_out.txt && sed -n '/after 100 steps/,/best arm/p' baseline_qtable.txt | tail -n +2 | diff - <(head -12 check_out.txt) && echo MATCH
