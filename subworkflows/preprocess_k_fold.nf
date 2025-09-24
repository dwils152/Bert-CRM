#!/usr/env/bin nextflow
include { PREPROCESS_CRMS       } from "./preprocess_crms.nf"


workflow PREPROCESS_K_FOLD {
    take:
        genome
        crm
        non_covered
        high_pval
        blacklist
        organism
        chrom_sizes
    main:
        partition_chroms(organism, chrom_sizes, 3)

}

process partition_chroms {
    label "Orion"
    publishDir "${params.results}/PREPROCESS_CHROM_FOLD"
    input:
        val(organism)
        path(chrom_sizes)
        val(k)
    output:
        
    script:
        """
        python ${params.scripts}/data_processing/partition_chroms.py \
            --organism ${organism} \
            --chrom_sizes ${chrom_sizes} \
            --k ${k}
        """
}