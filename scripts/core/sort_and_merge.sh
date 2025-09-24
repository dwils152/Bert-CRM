#!/bin/bash

export TMPDIR=/projects/zcsu_research1/dwils152/tmp
awk 'BEGIN {OFS="\t"} $4 == 1 {print $0}' $1 > pos.bed
sort-bed  --max-mem 12G pos.bed > pos_sorted.bed
bedtools merge -i pos_sorted.bed > sorted_and_merged_crms.bed