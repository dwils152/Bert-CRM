#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = channel.fromPath("${params.data}/mouse/mm10.fa")
mouse_crm = channel.fromPath("${params.data}/mouse/Mouse_CRMs_lte_1e-6.bed")
mouse_high_pval = channel.fromPath("${params.data}/mouse/Mouse_CRMs_gt_1e-6.bed")
mouse_non_covered = channel.fromPath("${params.data}/mouse/non-coverage.bed")

human_genome = channel.fromPath("${params.data}/human/hg38.fa")
human_crm = channel.fromPath("${params.data}/human/Human_CRMs_lte_1e-6.bed")
human_high_pval = channel.fromPath("${params.data}/human/Human_CRMs_gt_1e-6.bed")
human_non_covered = channel.fromPath("${params.data}/human/non-coverage.bed")

test_dataset = channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/mouse/labels_chunk/chunk_99.fa.csv")

process mask_genome {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy' 
    input:
        path genome
        path non_covered
        path high_pval
        val organism
    output:
        path "*_masked.fa"
    script:
        """
        bedtools maskfasta -fi ${genome} -bed ${non_covered} -fo tmp.fa
        bedtools maskfasta -fi tmp.fa -bed ${high_pval} -fo ${genome}_masked.fa
        """
}

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
        python ${params.scripts}/segment_genome.py ${genome}
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
    publishDir "${params.publish_dir}/${organism}/labels_chunk", mode: 'copy'
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

process cat_labels {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}", mode: 'copy'
    input:
        path chunked_labels
        val organism
    output:
        path "all_labels.csv"
    script:
        """
        cat *.csv > all_labels.csv
        """

}

process predict {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/prediction", mode: 'copy'
    input:
        path test_data
        val organism
    output:
        path "predictions_rank_*.npy"
        path "true_labels_rank_*.npy"
        path "model.pth"
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        ${params.scripts}/train.py ${test_data}
        """
}

process predict_no_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/prediction_no_crf", mode: 'copy'
    input:
        path test_data
        val organism
    output:
        path "predictions_rank_*.npy"
        path "true_labels_rank_*.npy"
        path "proba_rank_*.npy"
        path "model.pth"
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        ${params.scripts}/train_no_crf.py ${test_data}
        """
}


workflow {
        mask_genome(mouse_genome, mouse_non_covered, mouse_high_pval, "mouse")
        remove_scaffolds(mask_genome.out, "mouse")
        split_genome(remove_scaffolds.out, "mouse")
        chunk_for_labels(split_genome.out, "mouse")
        generate_labels(chunk_for_labels.out.flatten(), mouse_crm.first(), "mouse")
        cat_labels(generate_labels.out.collect(), "mouse")
        predict(test_dataset, "mouse")
        predict_no_crf(test_dataset, "mouse")
        
}