#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_genome = "${params.data}/mouse/mm10.fa"
human_genome = "${params.data}/human/hg38.fa"
//chicken_genome = "/projects/zcsu_research1/dwils152/Bert-CRM/data/chicken/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna"
rat_genome = "/projects/zcsu_research1/dwils152/Bert-CRM/data/mammals/rn6.fa"
fish_genome = "/projects/zcsu_research1/dwils152/Bert-CRM/data/mammals/danRer11.fa"
cow_genome = "/projects/zcsu_research1/dwils152/Bert-CRM/data/mammals/bosTau9.fa"
//human_no_crf_model = "/projects/zcsu_research1/dwils152/Bert-CRM/all_results/supervised_results_bug_fix/human/train_no_crf/model.pth"
human_no_crf_model = '/projects/zcsu_research1/dwils152/Bert-CRM/all_results/supervised_results_v2/human/train_no_crf/model.pth'


mouse_train_crf_dir = "${workflow.projectDir}/../supervised_results/mouse/train_crf"
mouse_train_no_crf_dir = "${workflow.projectDir}/../supervised_results/mouse/train_no_crf"
human_train_crf_dir =  "${workflow.projectDir}/../supervised_results/human/train_crf"
human_train_no_crf_dir =  "${workflow.projectDir}/../supervised_results/human/train_no_crf"





process split_genome {
    label "Orion"
    publishDir "${params.publish_dir}/split_genomes", mode: 'copy' 
    input:
        tuple path(genome), val(organism)
        //path(genome)
        //val(organism)
    output:
        tuple path("*.fa"), val(organism)
    script:
        """
        python ${params.scripts}/data_processing/segment_genome.py ${genome}
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
        ${params.scripts}/core/predict.py --model_path ${model} --data_path ${split_genome} --use_crf --is_distributed
        """
}

process predict_no_crf {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/no_crf_${pred_ord}_pred", mode: 'copy'
    input:
       tuple path(split_genome), val(pred_org), path(model)//, val(model_org) 
    output:
        "*"
    script:
        """
        export TMPDIR=/scratch/dwils152/tmp
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=1
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=2 \
        ${params.scripts}/core/predict.py --model_path ${model} --data_path ${split_genome} --is_distributed
        """
}

process predict_supervised {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}/train_non-shuf_test_shuf", mode: 'copy'
    input:
        path(test_data)
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
        --nproc_per_node=1 \
        ${params.scripts}/core/predict_supervised.py --model_path ${model} --data_path ${test_data}
        """

}


workflow {

    /*
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

    //predict_crf(crf)
    predict_no_crf(no_crf)
    */

    //shuff_data = Channel.fromPath(shuff_data)
    //non_shuf_model = Channel.fromPath(non_shuf_model)

    //shuff_data.view()
    //non_shuf_model.view()

    //predict_supervised(shuff_data, non_shuf_model)



    genome = Channel.from(tuple(rat_genome, "rat"), tuple(fish_genome, "fish"), tuple(cow_genome, "cow"))
    split_genome_ch = genome | split_genome

    //Add the model to the split_genome_ch tuples

    split_genome_ch.map{ genome, org -> tuple(genome, org, human_no_crf_model)}.set{ predict_input}
    predict_no_crf(predict_input) 

    

}