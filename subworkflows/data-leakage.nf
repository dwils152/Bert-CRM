#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

split_genomes = Channel.fromPath('./data/split_genomes/mm10_1000_0_1.0.fa')

process shuffle_dataset {
    label 'BigMem'
    publishDir "${params.publish_dir}", mode: 'copy'
    input:
        path fasta_1000bp
    output:
        path '*.fa'
    script:
        """
        python ${params.scripts}/data_processing/shuffled_and_n_last.py ${fasta_1000bp}
        """
}

process vsearch {
    label 'Clust'
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path shuf_fasta_1000bp
    output:
        path 'centroids.fasta'
        path 'clusters.uc'
    script:
    """
    vsearch --cluster_fast ${shuf_fasta_1000bp} --id 0.35 --centroids centroids.fasta --uc clusters.uc --threads 16

    """
}

process meshcluster {
    label 'Clust'
    publishDir "${params.publish_dir}", mode: 'copy'
    input:
        path shuf_fasta_1000bp
    output:
        path 'clusters.txt'
    script:
    """
    meshclust2 --id 0.75 ${shuf_fasta_1000bp} > clusters.txt
    """
}

workflow {
    
    shuffle_dataset(split_genomes)
    vsearch(shuffle_dataset.out)
    //meshcluster(shuffle_dataset.out)


}