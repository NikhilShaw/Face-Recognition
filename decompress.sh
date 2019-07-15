#!/bin/sh
for i in *.tar.gz; do
    tarname=`basename "$i" .gz`
    uncompress "$i"
    gzip "$tarname"
    echo "done"	
done
