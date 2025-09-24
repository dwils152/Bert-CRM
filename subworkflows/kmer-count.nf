#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

//mouse_crm = "${params.data}/mouse/fasta/Mouse_CRMs_lte_0.05_UPPER.fa"
//human_crm = "${params.data}/human/fasta/Human_CRMs_lte_0.05_UPPER.fa"
//mouse_ncrm = "${params.data}/mouse/fasta/NCRMs_UPPER.fa"
//human_ncrm = "${params.data}/human/fasta/NCRMs_UPPER.fa"
mouse_crm = "${params.data}/mouse/fasta/NCRMs_shuffled_1st.fa"
human_crm = "${params.data}/human/fasta/shuffled/NCRMs_UPPER_shuffled_1st.fa"

params.publish_dir = "/projects/zcsu_research1/dwils152/Bert-CRM/results"

process count_kmers {
    cache false
    label "Orion"
    publishDir "${params.publish_dir}/${organism}/${k}-mer", mode: 'copy'
    input:
        tuple val(organism), val(crm), val(k)
    output:
        tuple val(organism), val(k), path("output_${k}.jf")
    script:
        """
        jellyfish count -m ${k} -s 12G -t 16 -o output_${k}.jf ${crm}
        echo "Debugging Info:"
        ls -lah
        echo "Publishing to: ${params.publish_dir}/${organism}/${k}-mer"
        """
}

process dump_kmers {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}/${k}-mer", mode: 'copy'
    input:
        tuple val(organism), val(k), path(jf_binary)
    output:
        tuple val(organism), val(k), path("output_${k}.txt")
    script:
        """
        jellyfish dump -c ${jf_binary} > output_${k}.txt
        """
}

process plot_kmers {
    label "BigMem"
    publishDir "${params.publish_dir}/${organism}/${k}-mer", mode: 'copy'
    input:
        tuple val(organism), val(k), path(jf_text)
    output:
        path("output_${k}-${organism}.png")
    script:
        """
        python ${params.scripts}/kmer/plot_kmers.py ${jf_text} ${k} ${organism}
        """
}

process panel_images  {
    label "Orion"
    publishDir "${params.publish_dir}/${organism}/${k}-mer", mode: 'copy'
    input:
        path(images)
    output:
        tuple val(organism), val(k), path("output_${k}-${organism}.png")
    script:
        """
        python ${params.scripts}/kmer/plot_panel.py ${images}
        """
}


workflow {

    crms = Channel.from(tuple("mouse", mouse_crm), tuple("human", human_crm))
    k = Channel.from(1..7)
    crms.combine(k).set{ inputs }
    inputs | count_kmers | dump_kmers | plot_kmers


}