#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

include { PREPROCESS_CRMS       } from "./subworkflows/preprocess_crms.nf"
include { PREPROCESS_K_FOLD     } from "./subworkflows/preprocess_k_fold.nf"
include { SOURMASH_CLUSTER      } from "./subworkflows/sourmash_cluster.nf"   
include { MMSEQS_CLUSTER        } from "./subworkflows/mmseqs2_clustering.nf"
include { PREPROCESS_SINGLE_NT  } from "./subworkflows/preprocess_single_nt.nf"
include { TRAIN_CRMS_MODEL      } from "./subworkflows/supervised_cls_token.nf"
include { CRM_LOCATION_BIAS     } from "./subworkflows/crm_location_bias.nf"
include { CROSS_SPECIES_PRED    } from "./subworkflows/cross_species_pred.nf"


workflow  {

    // fasta = Channel.fromPath(
    //     "/projects/zcsu_research1/dwils152/Bert-CRM/results/Nt-Transformer/Worm/ce11_masked_no_scaffolds_upper.chunks.lt_0.5_n.fa"
    // )

    // genome = Channel.fromPath([
    //     //"/projects/zcsu_research1/dwils152/Bert-CRM/data/human/fasta/hg38.fa",
    //     //"/projects/zcsu_research1/dwils152/Bert-CRM/data/mouse/fasta/mm10.fa",
    //     // "/projects/zcsu_research1/dwils152/Bert-CRM/data/fly/fasta/dm6.fa",
    //     // "/projects/zcsu_research1/dwils152/Bert-CRM/data/worm/fasta/ce11.fa"
    // ])

    // SOURMASH_CLUSTER( genome )

    // PREPROCESS_CRMS(
    //     "${params.genome}",
    //     "${params.crms}",
    //     "${params.high_pval}"j,
    //     "${params.non_covered}",
    //     "${params.blacklist}",
    //     "${params.organism}"
    // )

    PREPROCESS_K_FOLD(
        "${params.genome}",
        "${params.crms}",
        "${params.high_pval}",
        "${params.non_covered}",
        "${params.blacklist}",
        "${params.organism}",
        "${params.chrom_sizes}"
    )

        // PREPROCESS_SINGLE_NT(
        //     "${params.genome}",
        //     "${params.crms}",
        //     "${params.high_pval}",
        //     "${params.non_covered}",
        //     "${params.blacklist}",
        //     "${params.organism}"
        // )

        // CRM_LOCATION_BIAS(
        //     "${params.n_bedfile_merged}",
        //     "${params.crms}",
        //     "${params.ncrms}"
        // )

        // fasta = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Fly/dm6_masked_no_scaffolds_upper.chunks.lt_0.5_n.fa")
        // scores = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Fly/labels.mmap")
        // deepspeed_config = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/ds_config_zero2.json")
        // TRAIN_CRMS_MODEL(fasta, scores, deepspeed_config)

        // Use MOUSE MODEL BUT HUMAN LABELs
        // model = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Worm/model")
        // lora_adapter = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Worm/LoRA")
        // fasta = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Human/hg38_masked_no_scaffolds_upper.chunks.lt_0.5_n.fa")
        // labels = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/results/Human/labels.mmap")
        // deepspeed_config = Channel.fromPath("/projects/zcsu_research1/dwils152/Bert-CRM/ds_config_zero2.json")
        
        // CROSS_SPECIES_PRED(
        //     fasta,
        //     labels,
        //     deepspeed_config,
        //     model,
        //     lora_adapter
        // )





}