#!/usr/bin/env nextflow
nextflow.enable.dsl=2

workflow MINHASH_CLUSTER {
    take:
        fasta
    main:
        minhash(fasta)

        pairwise_distance(minhash.out.sketch)

        cluster_map(pairwise_distance.out.pairwise_tsv)

}

process minhash {
    label "Multithread"
    publishDir "${params.results}/MinHash"
    input:
        path(fasta)
    output:
        path("${fasta}.msh"), emit: sketch
    script:
        """
        seqkit replace -p "[: ]" -r "_" ${fasta} > cleaned.fa
        mash sketch -i -p 32 -o "${fasta}.msh" cleaned.fa
        """
}

process pairwise_distance {
    label "Multithread"
    publishDir "${params.results}/MinHash"
    input:
        path(sketch) 
    output:
        path("pairwise.tab"), emit: pairwise_tsv
    script:
        """
        mash dist -p 32 ${sketch} ${sketch} > pairwise.tab
        """
}


process cluster_map {
    label "Orion"
    publishDir "${params.results}/MinHash"
    input:
        path(pairwise)
    output:
        path("clustermap.png")
    script:
        """
        python ${params.scripts}/clustering/plot_clustermap.py ${pairwise} clustermap.png
        """
}
