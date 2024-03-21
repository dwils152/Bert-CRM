#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = "${params.data}/mouse/mm10.fa"
mouse_crm = "${params.data}/mouse/Mouse_CRMs_lte_0.05.bed"
//mouse_high_pval = "${params.data}/mouse/Mouse_CRMs_gt_1e-6.bed"
mouse_non_covered = "${params.data}/mouse/non-coverage.bed"

human_genome = "${params.data}/human/hg38.fa"
human_crm = "${params.data}/human/Human_CRMs_lte_0.05.bed"
//human_high_pval = "${params.data}/human/Human_CRMs_gt_1e-6.bed"
human_non_covered = "${params.data}/human/non-coverage.bed"

test_dataset = channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/mouse/labels_chunk/chunk_99.fa.csv")

        //#bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo tmp.fa
        //#bedtools maskfasta -fi tmp.fa -bed ${high_pval} -fo \$outfile

process mask_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), path(non_covered), val(organism)
    output:
        tuple path("*_masked.fa"), val(organism)
    script:
        """
        genome=${genome}
        outfile=\${genome%.*}_masked.fa
        bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo \$outfile
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
        python ${params.scripts}/remove_scaffold.py ${genome}
        """
}

process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*.fa"), val(organism)
    script:
        """
        python ${params.scripts}/segment_genome.py ${genome}
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
        ${params.scripts}/chunk.sh ${split_genome} 6500
        """
}

process generate_labels {
    label "Orion"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/labels_chunk", mode: 'copy'
    input:
        tuple val(organism), path(crm), path(chunk)
    output:
        tuple (path "*.csv"), val(organism)
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python ${params.scripts}/generate_csv.py ${chunk} ${crm}
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
        ${params.scripts}/train.py --use_crf --data_path ${all_labels}
        """
}

process train_no_crf {
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
        --nproc_per_node=4 \
        ${params.scripts}/train.py --data_path ${all_labels}
        """
}


workflow {

        input_data = Channel.from(tuple(mouse_genome, mouse_non_covered,  "mouse"),
                                  tuple(human_genome, human_non_covered,  "human"))

        input_data | mask_genome | remove_scaffolds | split_genome | chunk_for_labels

        org_crm = Channel.from(tuple("mouse", mouse_crm), tuple("human", human_crm))
                         .join(chunk_for_labels.out)
                         .flatMap{ org, crm, chunk_list -> 
                            chunk_list.collect{ chunk -> tuple(org, crm, chunk) } }

        org_crm | generate_labels

        labels = generate_labels.out.groupTuple(by: 1) | cat_labels
        
        train_crf(labels)
        train_no_crf(labels)
}