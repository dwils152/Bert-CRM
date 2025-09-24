#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow PREPROCESS_CRMS {
    take:
        genome
        crm
        non_covered
        high_pval
        blacklist
        organism
    main:
        input = Channel.of(
            tuple(
                file(genome),
                file(non_covered),
                file(high_pval),
                file(blacklist),
                organism
            )
        )
        mask_genome(input)
        remove_scaffolds(mask_genome.out)
        genome_to_upper(remove_scaffolds.out)
        tokenize_genome(genome_to_upper.out)
        filter_by_n(tokenize_genome.out, 0.5)
        intersect_tokens(filter_by_n.out, crm)
        write_scores(intersect_tokens.out)
    emit:
        tokenize_genome.out[0]
        write_scores.out

}

process mask_genome {
    // Mask the non-covered, low-signficant, and black listed regions 
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), path(non_covered), path(high_pval), path(blacklist), val(organism)
    output:
        tuple path("*_masked.fa"), val(organism)
    script:
        """
        genome=${genome}
        outfile=\${genome%.*}_masked.fa
        bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo tmp.fa
        bedtools maskfasta -fi tmp.fa -bed ${blacklist} -fo tmp2.fa
        bedtools maskfasta -fi tmp2.fa -bed ${high_pval} -fo \$outfile
        """
}

process remove_scaffolds {
    // Remove unassembled chromosome scaffolds
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*_no_scaffolds.fa"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/remove_scaffold.py ${genome}
        """
}

process genome_to_upper {
    // Remove unassembled chromosome scaffolds
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*_upper.fa"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/to_upper.py ${genome} ${genome.baseName}_upper.fa
        """
}

process tokenize_genome {
    label "Tokenize"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*_upper.chunks.fa"), path("*_upper.chunks.bed"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/tokenize_genome_single-thread.py -i ${genome}
        """
}

process filter_by_n {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        tuple path(fasta_chunks), path(bed_chunks), val(organism)
        val(proportion)
    output:
        tuple path("*.chunks.lt_${proportion}_n.fa"), path("*.chunks.lt_${proportion}_n.bed"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/filter_by_n.py \
            --fasta ${fasta_chunks} \
            --bed ${bed_chunks} \
            --proportion_n ${proportion}
        """
}

process intersect_tokens {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(fasta_chunks), path(tokens_bed), val(organism)
        path(crms)
    output:
        tuple path("annotated.bed"), val(organism)
    script:
        """
        bgzip -c ${crms} > crms.bed.gz
        tabix -p bed crms.bed.gz

        bgzip -c ${tokens_bed} > tokens.bed.gz
        tabix -p bed tokens.bed.gz

        bedtools intersect -a tokens.bed.gz -b crms.bed.gz -c > annotated.bed
        """
}

process write_scores {
    label "Orion"
    // Intersect the single nucleotide intervals with the CRMs
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(annotated_bed), val(organism)
    output:
        tuple path("labels.mmap"), path("id_list.txt"), val(organism)
    script:
        """
        ${params.scripts}/data_processing/write_scores_2047 ${annotated_bed} labels.mmap id_list.txt
        """
}
