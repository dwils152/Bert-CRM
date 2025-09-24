#!/usr/bin/env nextflow
nextflow.enable.dsl=2


workflow SOURMASH_CLUSTER {
    take:
        genome
    main:

        split_genomes = split_genome(genome)

        split_genomes.collectFile(
            name: "${params.results}/SourMash/genomes/all_genomes.upper.1kb.fa"
        ).set{ all_genomes }


        sketch(all_genomes)

        // multisearch(sketch.out.sigs)

        //clustermap(multisearch.out)

}

process split_genome {
    label "Orion"
    publishDir "${params.results}/SourMash"
    input:
        path(genome)
    output:
        path("${genome.baseName}.upper.1kb.fa")
    script:
        """
        seqkit seq -u ${genome} > ${genome.baseName}.upper
        seqkit sliding -s 1000 -W 1000 ${genome.baseName}.upper > \
            ${genome.baseName}.upper.1kb.fa

        rm ${genome.baseName}.upper
        echo "hello"
        """
}

process sketch {
    label "Orion"
    publishDir "${params.results}/SourMash"
    input:
        path(fasta)
    output:
        path("*.sig.zip"), emit: sigs
    script:
        """
        sourmash sketch dna -p k=31,scaled=1000,abund \
        --singleton ${fasta} -o ${fasta}.fa.sig.zip
        """
}

process build_index {
    label "Orion"
    publishDir "${params.results}/SourMash"
    input:
        path(sigs)
    output:
        path("sourmash.rocksdb")
    script:
        """
        sourmash index sourmash.rocksdb ${sigs} -F rocksdb
        """
}

process multisearch {
    label "Multithread"
    publishDir "${params.results}/SourMash/Search"
    input:
        path(query)
    output:
        path("results.csv")
    script:
        """
        sourmash scripts multisearch \
            ${query} ${query} -o results.csv --cores 32
        """
}

process clustermap {
    label "DynamicAlloc"
    publishDir "${params.results}/SourMash/Clustermap"
    input:
        path(results)
    output:
        path("*.png")
    script:
        """
        sourmash scripts clustermap1 ${results} \
            -o clustermap1.sketches.png
        """
}


/*
        sourmash search \
            --num-results 150 \
            --ignore-abundance \
            --threshold 0.0 \
            ${query} ${db} > search.csv
            */