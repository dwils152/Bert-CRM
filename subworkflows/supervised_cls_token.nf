#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow TRAIN_CRMS_MODEL {
    take:
        input
        labels
        deepspeed_config
    main:
        train(input, labels, deepspeed_config)
        eval(
            input,
            labels,
            deepspeed_config,
            train.out
        )
        plot_results(eval.out)
}

process train {
    label "GPU"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path(fasta)
        path(labels)
        path(deepspeed_config)
    output:
        tuple path("model"), path("test_indices.pt"), path('LoRA')
    script:
        """
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        deepspeed --num_gpus 4 ${params.scripts}/core/run_models.py \
            --fasta  ${fasta} \
            --labels ${labels} \
            --learning_rate 1e-5 \
            --num_epochs 3 \
            --run_name train \
            --deepspeed ${deepspeed_config}
        """
}

process eval {
    label "GPU"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path(fasta)
        path(labels)
        path(deepspeed_config)
        tuple path(model), path(test_indices), path(lora)
    output:
        path("*")
    script:
        """
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        deepspeed --num_gpus 4 ${params.scripts}/core/deepspeed-predict.py \
            --fasta  ${fasta} \
            --labels ${labels} \
            --model_checkpoint ${model} \
            --test_indices_path ${test_indices} \
            --lora_dir ${lora} \
            --run_name test \
            --deepspeed ${deepspeed_config}
        """
}

process plot_results {
    label "BigMem"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path(model_output)
    output:
        path("*.png")
    script:
        """
        python ${params.scripts}/core/plot_eval.py
        """
}

