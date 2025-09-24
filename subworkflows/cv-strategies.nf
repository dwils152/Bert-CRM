#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2


process generate_embeddings {
    label "GPU"
    publishDir "${params.publish_dir}/${partition}", mode: 'copy' 
    input:
        tuple val(partition), path(fasta)
    output:
        "embeddings.npy"
    script:
        """
        export TOKENIZERS_PARALLELISM=false
        python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        ${params.scripts}/core/get_embeddings.py --data_path ${fasta}
        """
}


workflow {

    train = "${params.data}/split_genomes/vsearch/human/vsearch_0.35/sample/train_set.fa"
    val = "${params.data}/split_genomes/vsearch/human/vsearch_0.35/sample/val_set.fa"
    test = "${params.data}/split_genomes/vsearch/human/vsearch_0.35/sample/test_set.fa"
    input = Channel.from(tuple('train', train), tuple('val', val), tuple('test', test))
    input.view()
    generate_embeddings(input)

}