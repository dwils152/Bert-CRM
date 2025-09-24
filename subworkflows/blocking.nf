#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

tokenizer = "${params.data}/tokenizer.json"
split_genome = "${params.home}/all_results/supervised_results_bug_fix/human/all_labels.csv"

process generate embeddings {
    label "Orion"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path fasta
    output:
        "*"
    script:
        """
        
        """
}




workflow {



}