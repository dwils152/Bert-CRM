#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = "${params.data}/mouse/mm10.fa"
human_genome = "${params.data}/human/hg38.fa"

mouse_train_crf_dir = "${workflow.projectDir}/supervised_results/mouse/train_crf"
mouse_train_no_crf_dir = "${workflow.projectDir}/supervised_results/mouse/train_no_crf"
human_train_crf_dir =  "${workflow.projectDir}/supervised_results/human/train_crf"
human_train_no_crf_dir =  "${workflow.projectDir}/supervised_results/human/train_no_crf"

mouse_crf_model = "${mouse_train_crf_dir}/model.pth"
mouse_no_crf_model = "${mouse_train_no_crf_dir}/model.pth"
human_crf_model = "${human_train_crf_dir}/model.pth"
human_no_crf_model = "${human_train_no_crf_dir}/model.pth"

//test_data = 

process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/split_genomes", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*.fa"), val(organism)
    script:
        """
        python ${params.scripts}/segment_genome.py ${genome}
        """
}

process predict_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/crf_${pred_ord}_pred_${model_org}", mode: 'copy'
    input:
        tuple path(split_genome), val(pred_org), path(model), val(model_org)
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
        ${params.scripts}/predict.py --model_path ${model} --data_path ${split_genome} --use_crf
        """
}

process predict_no_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/no_crf_${pred_ord}_pred_${model_org}", mode: 'copy'
    input:
       tuple path(split_genome), val(pred_org), path(model), val(model_org) 
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
        ${params.scripts}/predict.py --model_path ${model} --data_path ${split_genome}
        """
}


workflow {

    genome = Channel.from(tuple(mouse_genome, "mouse"), tuple(human_genome, "human"))
    models = Channel.from(
        tuple(mouse_crf_model, "mouse"),
        tuple(mouse_no_crf_model, "mouse"),
        tuple(human_crf_model, "human"),
        tuple(human_no_crf_model, "human"))

    split_genome_ch = genome | split_genome
    model_inputs = split_genome_ch.combine(models).filter{ it[1] != it[3] }

    crf = model_inputs.filter{ it[2].contains("train_crf") }
    no_crf = model_inputs.filter{ it[2].contains("train_no_crf") }

    predict_crf(crf)
    predict_no_crf(no_crf)

    

}