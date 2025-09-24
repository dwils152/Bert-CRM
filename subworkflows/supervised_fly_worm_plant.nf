#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

fly_genome = "${params.data}/fly/fasta/dm3.fa"
fly_crm = "${params.data}/fly/bed/ALL_CRMs_300.crm.bed.clean"
fly_non_covered = "${params.data}/fly/bed/non-coverage.bed"

worm_genome = "${params.data}/worm/fasta/ce11.fa"
worm_crm = "${params.data}/worm/bed/ALL_CRMs_300.crm.bed.clean"
worm_non_covered = "${params.data}/worm/bed/non-coverage.bed"

// plant_genome = "${params.data}/plant/fasta/hg38.fa"
// plant_crm = "${params.data}/plant/bed/ALL_CRMs_300.crm.bed.clean"
// plant_non_covered = "${params.data}/plant/bed/non-coverage.bed"

//#bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo tmp.fa
//#bedtools maskfasta -fi tmp.fa -bed ${high_pval} -fo \$outfile
//#bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo \$outfile

process mask_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), path(non_covered), path(crm), val(organism)
    output:
        tuple path("*_masked.fa"), val(organism)
    script:
        """
        genome=${genome}
        outfile=\${genome%.*}_masked.fa
        mv \${genome} \$outfile
        #bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo tmp.fa
        #bedtools maskfasta -fi tmp.fa -bed ${crm} -fo \$outfile
        """
}

process remove_scaffolds {
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

// Softmasked sequences are turned uppercase in the segment_genome script
process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*.fa"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/segment_genome.py ${genome}
        """
}

process chunk_for_labels {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        tuple path(split_genome), val(organism)
    output:
        tuple val(organism), path("fasta_chunks/*.fa")
    script:
        """
        ${params.scripts}/data_processing/chunk.sh ${split_genome} 6500
        """
}

process generate_labels {
    label "DynamicAlloc"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/labels_chunk", mode: 'copy'
    input:
        tuple val(organism), path(crm), path(chunk)
    output:
        tuple (path "*.csv"), val(organism)
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python ${params.scripts}/data_processing/generate_csv_fast.py ${chunk} ${crm}
        """
}

process cat_labels {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        tuple path(all_labels), val(organism)
    output:
        tuple path("all_labels.csv"), val(organism)
    script:
        """
        cat *.csv > all_labels.csv
        """
}

process train_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/train_crf", mode: 'copy'
    input:
        tuple path(all_labels), val(organism)
    output:
        path "predictions_rank_*.npy"
        path "true_labels_rank_*.npy"
        path "model.pth"
        val organism
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        ${params.scripts}/core/train.py --use_crf --data_path ${all_labels}
        """
}

process train_no_crf {
    cache false
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/train_no_crf", mode: 'copy'
    input:
        tuple path(all_labels), val(organism)
    output:
        path "predictions_rank_*.npy"
        path "true_labels_rank_*.npy"
        path "proba_rank_*.npy"
        path "model.pth"
        val organism
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=2 \
        ${params.scripts}/core/train.py --data_path ${all_labels}
        """
}


workflow {

        
        input_data = Channel.from(tuple(fly_genome, fly_non_covered, fly_crm,  "fly"),
                                 tuple(worm_genome, worm_non_covered, worm_crm, "worm"))

        input_data | mask_genome | remove_scaffolds | split_genome | chunk_for_labels

        org_crm = Channel.from(tuple("fly", fly_crm), tuple("worm", worm_crm))
                         .join(chunk_for_labels.out)
                         .flatMap{ org, crm, chunk_list -> 
                            chunk_list.collect{ chunk -> tuple(org, crm, chunk) } }

        org_crm | generate_labels
        labels = generate_labels.out.groupTuple(by: 1) | cat_labels
        
        //train_crf(labels)
        train_no_crf(labels)
        
        
  }