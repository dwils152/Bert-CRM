#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow CLUSTER_SEQS {
    take:
        fasta
    main:
        create_database(fasta)
        cluster_seqs(fasta, create_database.out)
}

process create_database {
    label "Orion"
    publishDir "${params.publish_dir}/clustering", mode: 'copy' 
    input:
        path(fasta)
    output:
        path("inputDB")
    script:
        """
        mmseqs createdb ${fasta} inputDB
        """
}

process linclust {
    label "BigMem"
    publishDir "${params.publish_dir}/clustering", mode: 'copy' 
    input:
        path(fasta)
    output:
        path("clusterDB")
    script:
        """
        mmseqs linclust inputDB clusterDB tmp \
            --min-seq-id 0.7 \
            -c 0.8 \
            --cov-mode 1 \
            --kmer-per-seq 100 \
            --threads 64 \
            --split-memory-limit 75G
        """
}