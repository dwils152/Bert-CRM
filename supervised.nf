#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = channel.fromPath("${params.data}/mouse/mm10.fa")
mouse_crm = channel.fromPath("${params.data}/mouse/Mouse_CRMs_lte_1e-6.bed")

human_genome = channel.fromPath("${params.data}/human/hg38.fa")
human_crm = channel.fromPath("${params.data}/human/hg38_crm.bed")

process remove_scaffolds {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        path genome
        val organism
    output:
        path "*_no_scaffolds.fa"
    script:
        """
        python ${params.scripts}/remove_scaffold.py ${genome}
        """
}

process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        path genome
        val organism
    output:
        path "*.fa"
    script:
        """
        python ${params.scripts}/preprocessor.py ${genome}
        """
}

process chunk_for_labels {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        path split_genome
        val organism
    output:
        path "fasta_chunks/*.fa"
    script:
        """
        ${params.scripts}/chunk.sh ${split_genome} 6500
        """
}

process generate_labels {
    label "Orion"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        path chunk
        path crm
        val organism
    output:
        path "*.csv"
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python ${params.scripts}/generate_csv.py ${chunk} ${crm}
        """
}

workflow preprocess_data {
    main:
        remove_scaffolds(mouse_genome, "mouse")
        split_genome(remove_scaffolds.out, "mouse")
        chunk_for_labels(split_genome.out, "mouse")
        generate_labels(chunk_for_labels.out.flatten(), "mouse")
}