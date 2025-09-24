#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2


fly_genome = "/projects/zcsu_research1/dwils152/Bert-CRM/data/fly/fasta/dm3.fa"
fly_no_crf_model = "/projects/zcsu_research1/dwils152/Bert-CRM/all_results/results_spec/fly/train_no_crf/model.pth"

process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/split_genomes", mode: 'copy' 
    input:
        path(genome)
    output:
        tuple path("*.fa")
    script:
        """
        python ${params.scripts}/data_processing/segment_genome.py ${genome}
        """
}

process attribution {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/no_crf_${pred_ord}_pred_${model_org}", mode: 'copy'
    input:
       path(split_genome)
       path(model)
    output:
        "*"
    script:
        """
        export TMPDIR=/scratch/dwils152/tmp
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=1
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        ${params.scripts}/core/attribution.py --model_path ${model} --data_path ${split_genome}
        """
}

workflow {

    split_genome(Channel.fromPath(fly_genome))
    attribution(split_genome.out, Channel.fromPath(fly_no_crf_model))

    

}