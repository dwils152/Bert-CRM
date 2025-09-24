#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow PREDICT_CROSS_SPECIES {
    take:
        model
        input
        labels
        deepspeed_config
    main:
        predict_cross_species(input, labels, deepspeed_config)
}

process predict_cross_species {
    label "GPU"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path(model)
        path(fasta)
        path(labels)
        path(deepspeed_config)
    output:
        path("*")
    script:
        """
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        deepspeed --num_gpus 4 ${params.scripts}/core/deepspeed.py \
            --model ${model} \
            --fasta  ${fasta} \
            --labels ${labels} \
            --run_name multi_gpu_experiment \
            --deepspeed ${deepspeed_config}
        """

}

