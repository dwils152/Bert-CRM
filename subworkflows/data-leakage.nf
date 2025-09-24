#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

split_genomes = Channel.fromPath('./data/split_genomes/hg38_1000_0_1.0.fa')

process make_db {
    label 'Orion'
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path fasta_1000bp
    output:
        path 'db.*'
    script:
    """
    makeblastdb -in ${fasta_1000bp} -dbtype nucl -out db
    """
}

process blast {
    label 'BigMem'
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path db_files
        path fasta_1000bp
    output:
        path 'blast_out'
    script:
    """
    blastn -db db -query ${fasta_1000bp} -out blast_out
    """
}

process vsearch {
    label 'Clust'
    publishDir "${params.publish_dir}", mode: 'copy' 
    input:
        path fasta_1000bp
    output:
        path 'centroids.fasta'
        path 'clusters.uc'
    script:
    """
    vsearch --cluster_fast ${fasta_1000bp} --id 0.35 --centroids centroids.fasta --uc clusters.uc --threads 16

    """
}

workflow {
    
    //make_db(split_genomes)
    //blast(make_db.out, split_genomes)
    vsearch(split_genomes)

}