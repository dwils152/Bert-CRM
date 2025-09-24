#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

fasta = "/projects/zcsu_research1/dwils152/Bert-CRM/results/Human/hg38_masked_no_scaffolds_600_0_0.0_right.fa"

workflow SPANSEQ {
    take:
        fasta
    main:
        shuffle_fasta_order(fasta)
}

process shuffle_fasta_order {
    publishDir "${params.publish_dir}", mode: 'copy'
    input:
        path(fasta)
    output:
        path("${fasta}.shuf")
    script:
        """
        seqkit shuffle ${fasta} -o "${fasta}.shuf"
        """
}

