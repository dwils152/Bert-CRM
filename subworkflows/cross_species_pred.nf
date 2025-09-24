#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow CROSS_SPECIES_PRED {
    take:
        fasta
        labels
        deepspeed_config
        model
        lora
    main:
        predict(
            fasta,
            labels,
            deepspeed_config,
            model,
            lora
        )
        plot_results(predict.out)
}

process predict {
    label "GPU"
    publishDir "${params.publish_dir}/Worm_Model-Human_Labels", mode: 'copy' 
    input:
        path(fasta)
        path(labels)
        path(deepspeed_config)
        path(model)
        path(lora)
    output:
        path("*")
    script:
        """
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        deepspeed --num_gpus 4 ${params.scripts}/core/deepspeed-cross_spec.py \
            --fasta  ${fasta} \
            --labels ${labels} \
            --model_checkpoint ${model} \
            --lora_dir ${lora} \
            --run_name test \
            --deepspeed ${deepspeed_config}
        """
}

process plot_results {
    label "BigMem"
    publishDir "${params.publish_dir}/Worm_Model-Human_Labels", mode: 'copy' 
    input:
        path(model_output)
    output:
        path("*.png")
    script:
        """
        python ${params.scripts}/core/plot_eval.py
        """
}