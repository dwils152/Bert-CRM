#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

mouse_train_set = "${params.data}/split_genomes/vsearch/mouse/vsearch_0.35/train_set.fa"
mouse_val_set = "${params.data}/split_genomes/vsearch/mouse/vsearch_0.35/val_set.fa"
mouse_test_set = "${params.data}/split_genomes/vsearch/mouse/vsearch_0.35/test_set.fa"

//human_train_set = "/projects/zcsu_research1/dwils152/Bert-CRM/data/split_genomes/vsearch/human/vsearch_0.35/train_set.fa"
//human_val_set = "/projects/zcsu_research1/dwils152/Bert-CRM/data/split_genomes/vsearch/human/vsearch_0.35/val_set.fa"
//human_test_set = "/projects/zcsu_research1/dwils152/Bert-CRM/data/split_genomes/vsearch/human/vsearch_0.35/test_set.fa"


process sample_from_splits {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}_adversarial_val", mode: 'copy' 
    input:
        tuple val(organism), path(train_set), path(val_set), path(test_set)
    output:
        path "partition_*.fasta"
    script:
        """
        python sample_from_splits.py --train_set ${train_set} --val_set ${val_set} --test_set ${test_set}
        """
}


/*process predict_split_from_sets {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}_adversarial_val", mode: 'copy' 
    input:
    output:
    script:
        """
        """
}*/


workflow {

        input_data = Channel.from(tuple("mouse", mouse_train_set, mouse_val_set, mouse_test_set))


}