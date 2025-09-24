#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_labels = "/projects/zcsu_research1/dwils152/Bert-CRM/supervised_results_0.05/mouse/all_labels.csv"
mouse_data_splits = "${params.data}/split_genomes/vsearch/mouse/vsearch_0.35/dataset_splits.tsv"
human_labels = "/projects/zcsu_research1/dwils152/Bert-CRM/supervised_results_0.05/human/all_labels.csv"
human_data_splits= "${params.data}/split_genomes/vsearch/human/vsearch_0.35/dataset_splits.tsv"

process distribute_seqs {
    label "BigMem"
    publishDir "${params.publish_dir}", mode: 'copy'
    input:
        tuple path(labels), path(data_splits), val(organism)
    output:
        tuple path("train_split.csv"), path("val_split.csv"), path("test_split.csv"), val(organism)
    script:
        """
        python ${params.scripts}/bert_crm/data_processing/distribute_seqs.py ${labels} ${data_splits}
        """
}

process train_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/train_crf", mode: 'copy'
    input:
        tuple path(trainset), path(valset), path(testset), val(organism)
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
        ${params.scripts}/bert_crm/core/train.py --use_crf --use_splits --train_split ${trainset} --val_split ${valset} --test_split ${testset}
        """
}

process train_no_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/${organism}/train_no_crf", mode: 'copy'
    input:
        tuple path(trainset), path(valset), path(testset), val(organism)
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
        ${params.scripts}/bert_crm/core/train.py --use_splits --train_split ${trainset} --val_split ${valset} --test_split ${testset}
        """
}


workflow {

        input_data = Channel.from(tuple(mouse_labels, mouse_data_splits,  "mouse"),
                                  tuple(human_labels, human_data_splits,  "human"))


        distribute_seqs(input_data)
        train_crf(distribute_seqs.out)
        train_no_crf(distribute_seqs.out)
        

        
}