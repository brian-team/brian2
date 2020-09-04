#!/bin/bash
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
for a in `git log --pretty="%an" | sort | uniq`; do git log --author="$a" --pretty="%ai %an" | tail -n1; done
IFS=$SAVEIFS
