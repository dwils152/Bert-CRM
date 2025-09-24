#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

//human_dataset = "/projects/zcsu_research1/dwils152/Bert-CRM/data/human/fasta/Human_CRMs_lte_0.05_UPPER.fa"
shuf = "/projects/zcsu_research1/dwils152/Bert-CRM/data/human/fasta/sampled/Human_CRMs_lte_0.05_UPPER_500.fa"

process split_crms {
    label "Orion"
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path(crm)
    output:
       path "split_crm_pos_neg.fa" 
    script:
        """
        python "${params.scripts}/data_processing/split_crms.py" ${crm}
        """

}

process train {
    label "GPU"
    conda "/users/dwils152/.conda/envs/dna3"
    publishDir "${params.publish_dir}", mode: 'copy'
    input:
        path(split_crms)
    output:
        "*"
    script:
        """
        python ${params.scripts}/core/train_crm_split.py --fasta ${split_crms}
        """
}




workflow {

    crms = Channel.from(shuf)
    train(crms)
    


}