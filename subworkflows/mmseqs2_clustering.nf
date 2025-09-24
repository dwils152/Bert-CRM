#!/usr/bin/env nextflow
nextflow.enable.dsl=2

workflow MMSEQS_CLUSTER {
    take:
        fasta
    main:
        mmseqs_createdb(fasta)
        mmseqs_cluster(mmseqs_createdb.out_db)
        mmseqs_createtsv(mmseqs_createdb.out_db, mmseqs_cluster.out_clu)
}

process mmseqs_createdb {
    label "Multithread"
    publishDir "${params.results}/MMseqs", mode: 'copy'

    input:
        path(fasta)

    output:
        path("seqDB"), emit: out_db

    script:
        """
        seqkit replace -p "[: ]" -r "_" ${fasta} > cleaned.fa
        mmseqs createdb cleaned.fa seqDB
        """
}

process mmseqs_cluster {
    label "Multithread"
    publishDir "${params.results}/MMseqs", mode: 'copy'

    input:
        path(seqDB)

    output:
        path("clusterRes"), emit: out_clu

    script:
        """
        mmseqs cluster seqDB clusterRes tmp --min-seq-id 0.9 -c 0.8 --threads 32
        """
}

process mmseqs_createtsv {
    label "Multithread"
    publishDir "${params.results}/MMseqs", mode: 'copy'

    input:
        path(seqDB)
        path(clusterRes)

    output:
        path("pairwise.tab"), emit: pairwise_tsv

    script:
        """
        mmseqs createtsv seqDB seqDB clusterRes pairwise.tab
        """
}
