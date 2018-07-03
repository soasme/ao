#!/usr/bin/env bash

set -e;

run () {
    if [ `ao $1.ao` == `cat $1.out` ]; then
        echo "$1 passed.";
    else
        echo "$1 failed.";
    fi
}

for testcase in `ls *.ao | awk -F'.' '{print $1}'`;
do
    echo $testcase
    run testcase
done
