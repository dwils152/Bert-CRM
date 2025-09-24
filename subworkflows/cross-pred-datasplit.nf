#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = "${params.data}/mouse/fasta/mm10.fa"
human_genome = "${params.data}/human/fasta/hg38.fa"
fly_genome = "${params.data}/fly/fasta/dm3.fa"
worm_genome = "${params.data}/worm/fasta/ce11.fa"

mouse_train_no_crf_dir = "/projects/zcsu_research1/dwils152/Bert-CRM/results_spec/mouse/train_no_crf"
human_train_no_crf_dir =  "/projects/zcsu_research1/dwils152/Bert-CRM/results_spec/human/train_no_crf"

fly_train_no_crf_dir =  "/projects/zcsu_research1/dwils152/Bert-CRM/results_spec/fly/train_no_crf"
worm_train_no_crf_dir =  "/projects/zcsu_research1/dwils152/Bert-CRM/results_spec/worm/train_no_crf"

mouse_no_crf_model = "${mouse_train_no_crf_dir}/model.pth"
human_no_crf_model = "${human_train_no_crf_dir}/model.pth"

fly_no_crf_model = "${fly_train_no_crf_dir}/model.pth"
worm_no_crf_model = "${worm_train_no_crf_dir}/model.pth"


process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/split_genomes", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
    output:
        tuple path("*.fa"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/segment_genome.py ${genome}
        """
}

// process predict_crf {
//     label "GPU"
//     conda "/users/dwils152/.conda/envs/dna3"
//     publishDir "${params.publish_dir}/crf_${pred_ord}_pred_${model_org}", mode: 'copy'
//     input:
//         tuple path(split_genome), val(pred_org), path(model), val(model_org)
//     output:
//         "*"
//     script:
//         """
//         export TMPDIR=/scratch/dwils152/tmp
//         export TOKENIZERS_PARALLELISM=false
//         export OMP_NUM_THREADS=1
//         python -m torch.distributed.run \
//         --nnodes=1 \
//         --nproc_per_node=4 \
//         ${params.scripts}/bert_crm/core/predict.py --model_path ${model} --data_path ${split_genome} --use_crf
//         """
// }

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
        ${params.scripts}/core/predict.py --model_path ${model} --data_path ${split_genome}
        """
}


workflow {

    genome = Channel.from(tuple(mouse_genome, "mouse"), tuple(human_genome, "human"), tuple(fly_genome, "fly"), tuple(worm_genome, "worm"))
    models = Channel.from(
        tuple(mouse_no_crf_model, "mouse"),
        tuple(human_no_crf_model, "human"),
        tuple(fly_no_crf_model, "fly"),
        tuple(worm_no_crf_model, "worm"))

    split_genome_ch = genome | split_genome

    //split_genome_ch.view()

    model_inputs = split_genome_ch.combine(models).filter{ it[1] != it[3] }

    //model_inputs.view()

    //crf = model_inputs.filter{ it[2].contains("train_crf") }
    no_crf = model_inputs.filter{ it[2].contains("train_no_crf") }

    //no_crf.view()

    //predict_crf(crf)
    predict_no_crf(no_crf)

    

}