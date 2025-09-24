#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2


workflow CRM_LOCATION_BIAS {
    take:
        n_bed
        crms
        ncrms
    main:
        distance_to_crm_ncrm(
            n_bed,
            crms,
            ncrms
        )
}

process distance_to_crm_ncrm {
    label "Orion"
    publishDir "${params.publish_dir}/${params.organism}", mode: 'copy' 
    input:
        path(n_bed)
        path(crms)
        path(ncrms)
    output:
        tuple path("crm_dist_n.bed"), path("ncrm_dist_n.bed")
    script:
        """
        bedtools closest -a ${n_bed} -b ${crms} -d > crm_dist_n.bed
        bedtools closest -a ${n_bed} -b ${ncrms} -d > ncrm_dist_n.bed
        """
}

process plot_distances {
    label "Orion"
    publishDir "${params.publish_dir}/${params.organism}", mode: 'copy' 
    input:
        tuple path(crm_dist), path(ncrm_dist)
    output:
        path("crm_ncrm_distance_from_n.png")
    script:
        """
         python ${params.scripts}/data_processing/distance_to_n.py ${crm_dist} ${ncrm_dist}
        """
}