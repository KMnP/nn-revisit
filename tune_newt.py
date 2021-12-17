"""
train newt with knn results as reg
"""
from launch import default_argument_parser
from tune import main as tune_main
OUTPUT_DIR = ""


def _main(feature, data_name):
    # construct args for each jobs
    default_outputdir = OUTPUT_DIR
    args_list = []
    cfg = "configs/linear/newt.yaml"

    # construct args for each jobs
    # base
    args_list.append(default_argument_parser().parse_args([
        "--config-file", cfg,
        "--train-type", "linear",
        "MODEL.TYPE", "linear_base", "RUN_N_TIMES", "1",
        "DATA.NAME", data_name, "DATA.FEATURE", feature,
        "OUTPUT_DIR", default_outputdir + "_base",
    ]))

    # joint
    for knn_coeff in ["0.1", "0.01", "0.001", "0.0001"]:
        args_list.append(default_argument_parser().parse_args([
            "--config-file", cfg, "--train-type", "linear",
            "MODEL.TYPE", "linear_joint", "DSTORE.RETURN_PROBS", "True",
            "DSTORE.LOSS", "True", "SOLVER.LOSS", "knn_reg",
            "MODEL.KNN_LAMBDA", knn_coeff,
            "RUN_N_TIMES", "1",
            "DATA.NAME", data_name, "DATA.FEATURE", feature,
            "OUTPUT_DIR", default_outputdir + "_joint_knnregloss",
        ]))
    print(f"will launch {len(args_list)} jobs")
    for args in args_list:
        try:
            tune_main(args)
        except Exception as e:
            print(e)


def main_feat(feature_name):
    data_names = [
        'ml_bio_raptor_utility_pole',
        'inat_non_species_intersex_mallards',
        'ml_age_coopers_hawk',
        'cct_bobcat_vs_empty_loc_43_night',
        'inat_unobserved_amanita_flavorubens_v_amanita_xanthocephala',
        'ml_tag_nest',
        'inat_non_species_diseased_zebra_finch',
        'ml_age_black_bellied_plover',
        'inat_observed_California_Sea_Lion_vs_Steller_Sea_Lion',
        'inat_unobserved_chloris_verticillata_v_chloris_cucullata',
        'cct_coyote_vs_empty_loc_43_night',
        'nabirds_species_classification_truswa_tunswa',
        'inat_observed_Allegheny_Mountain_Dusky_Salamander_vs_Dusky_Salamander',
        'ml_photo_rating_12_vs_45_v2',
        'nabirds_species_classification_brwhaw_reshaw',
        'inat_observed_Jelly_Ear_vs_Ear_fungus',
        'inat_unobserved_pinus_clausa_v_pinus_mugo',
        'inat_observed_Common_Grass_Yellow_vs_Three-spotted_Grass_Yellow',
        'ml_tag_copulation',
        'inat_observed_Belize_Crocodile_vs_American_Crocodile',
        'inat_unobserved_turdus_torquatus_v_turdus_atrogularis',
        'nabirds_species_classification_coohaw_shshaw',
        'nabirds_species_classification_semsan_wessan',
        'cct_opossum_vs_cat_loc_38_night',
        'nabirds_species_classification_amecro_comrav',
        'nabirds_species_classification_sursco_whwsco2',
        'inat_observed_Western_Grey_Kangaroo_vs_Eastern_Grey_Kangaroo',
        'inat_non_species_mating_danaus_plexippus',
        'inat_non_species_white_american_robin',
        'inat_unobserved_cladonia_squamosa_v_cladonia_portentosa',
        'ml_photo_rating_12_vs_45_v3',
        'ml_tag_egg',
        'inat_unobserved_podarcis_virescens_v_podarcis_guadarramae',
        'inat_unobserved_cuphea_aequipetala_v_cuphea_hyssopifolia',
        'nabirds_species_classification_bargol_comgol',
        'inat_observed_Eastern_Oyster_vs_Pacific_Oyster',
        'inat_non_species_mating_aligator_lizard',
        'inat_observed_Rough_Green_Snake_vs_Smooth_Greensnake',
        'nabirds_species_classification_casvir_plsvir',
        'inat_unobserved_armillaria_luteobubalina_v_armillaria_novae-zelandiae',
        'inat_observed_Southern_Black_Widow_vs_Western_Black_Widow',
        'inat_observed_Flea_Jumper_vs_Asiatic_Wall_Jumping_Spider',
        'inat_observed_Northern_Cinnabar_Polypore_vs_Cinnabar_Bracket',
        'ml_age_sanderling',
        'ml_tag_foraging_waterfowl',
        'cct_opossum_vs_raccoon_loc_100_night',
        'inat_observed_southern_cattail_vs_lesser_reedmace',
        'inat_non_species_mating_toxomerus_marginatus',
        'nabirds_species_classification_easmea_wesmea',
        "inat_observed_Orange_Jelly_Spot_vs_witch's_butter",
        'fgvcx_plant_pathology_healthy_vs_sick',
        'inat_unobserved_cortinarius_austrovenetus_v_cortinarius_archeri',
        'nabirds_species_classification_botgra_grtgra',
        'nabirds_species_classification_bkchum_rthhum',
        'ml_tag_back_of_camera',
        'nabirds_species_classification_blkvul_turvul',
        'inat_unobserved_emberiza_pusilla_v_emberiza_leucocephalos',
        'inat_observed_Brown-lipped_Snail_vs_White-lipped_Snail',
        'inat_unobserved_hippolais_icterina_v_hippolais_polyglotta',
        'ml_tag_field_notes_sketch',
        'nabirds_species_classification_buhvir_casvir',
        'inat_non_species_mating_bagrada_hilaris',
        'inat_unobserved_leucorrhinia_dubia_v_leucorrhinia_rubicunda',
        'ml_tag_molting_waterfowl',
        'inat_observed_Red_Belted_Conk_vs_Northern_Red_Belt',
        'inat_observed_Western_Mosquitofish_vs_Eastern_Mosquitofish',
        'inat_non_species_white_white_tailed_deer',
        'ml_age_bald_eagle',
        'inat_observed_Northern_Two-lined_Salamander_vs_Southern_Two-lined_Salamander',
        'inat_observed_Desert_Blonde_Tarantula_vs_Desert_Tarantula',
        'cct_cat_vs_bobcat_loc_105',
        'inat_unobserved_thysanotus_tuberosus_v_thysanotus_patersonii',
        'inat_non_species_mating_oncopeltus_fasciatus',
        'ml_tag_vocalizing',
        'inat_unobserved_scopula_umbilicata_v_scopula_ornata',
        'ml_photo_rating_12_vs_45_v4',
        'cct_opossum_vs_raccoon_loc_33_night',
        'inat_non_species_feather_california_scrub_jay_v_quail',
        'nabirds_species_classification_hergul_ribgul',
        'inat_unobserved_judolia_cordifera_v_judolia_cerambyciformis',
        'inat_non_species_black_eastern_gray_squirrel',
        'inat_non_species_dead_striped_skunk',
        'inat_observed_Snakeskin_Chiton_vs_California_Spiny_Chiton',
        'cct_squirrel_vs_rabbit_loc_38_day',
        'cct_rabbit_vs_bird_loc_61_day',
        'nabirds_species_classification_gloibi_whfibi',
        'nabirds_species_classification_greyel_lesyel',
        'inat_observed_Eastern_Meadowlark_vs_Western_Meadowlark',
        'nabirds_species_classification_comrav_fiscro',
        'inat_unobserved_lactarius_torminosus_v_lactarius_turpis',
        "inat_observed_Cross_Orbweaver_vs_Hentz's_Orbweaver",
        'nabirds_species_classification_bkcchi_carchi',
        'inat_unobserved_apodemus_sylvaticus_v_apodemus_agrarius',
        'ml_tag_non_bird',
        'nabirds_species_classification_houfin_purfin',
        'inat_unobserved_oudemansiella_mucida_v_oudemansiella_furfuracea',
        'nabirds_species_classification_eawpew_wewpew',
        'inat_non_species_deformed_beak',
        'inat_observed_Eastern_Cane_Toad_vs_Giant_Marine_Toad',
        'inat_unobserved_serinus_canaria_v_serinus_canicollis',
        'inat_unobserved_corvus_orru_v_corvus_sinaloae',
        'glc_low_development_vs_high_development',
        'ml_age_swainsons_hawk',
        'cct_opossum_vs_cat_loc_100_night',
        "inat_observed_Pacific_Banana_Slug_vs_Button's_Banana_Slug",
        'nabirds_species_classification_rensap_yebsap',
        'nabirds_species_classification_buhvir_plsvir',
        'inat_non_species_mating_terrapene_carolina',
        'ml_photo_rating_12_vs_45_v5',
        'nabirds_species_classification_barswa_cliswa',
        'ml_tag_watermark',
        'ml_tag_molting_raptors',
        'inat_non_species_tagged_swan',
        'inat_non_species_diseased_leaves',
        'glc_pasture_vs_crop',
        'inat_non_species_mating_hippodamia_convergens',
        'inat_non_species_dead_jackal',
        'nabirds_species_classification_cacgoo1_cangoo',
        'ml_age_sharp_shinned_hawk',
        'nabirds_species_classification_linspa_sonspa',
        'nabirds_species_classification_savspa_sonspa',
        'inat_observed_Common_Shiny_Woodlouse_vs_Rathke’s_Woodlouse',
        'ml_bio_has_red_eyes',
        'inat_unobserved_aceria_negundi_v_aceria_cephalonea',
        'ml_age_dunlin',
        'nabirds_species_classification_canvas_redhea',
        'ml_age_semipalmated_plover',
        'ml_tag_in_hand',
        'ml_bio_high_contrast',
        'ml_tag_carrying_food',
        'inat_non_species_mammal_species',
        'ml_bio_is_at_flower',
        'inat_non_species_mating_chauliognathus_pensylvanicus',
        'inat_unobserved_lampsilis_cardium_v_lampsilis_siliquoidea',
        'nabirds_species_classification_orcwar_tenwar',
        'ml_tag_multiple_species',
        'inat_unobserved_podarcis_liolepis_v_podarcis_bocagei',
        'inat_non_species_mating_harmonia_axyridis',
        'inat_unobserved_lanius_bucephalus_v_lanius_meridionalis',
        'ml_age_least_sandpiper',
        'inat_unobserved_diaea_dorsata_v_diaea_ambara',
        'glc_woody_wetlands_vs_emergent_herbaceous_wetlands',
        'inat_observed_Southern_Cinnabar_Polypore_vs_Cinnabar_Bracket',
        'inat_unobserved_phaeophyscia_orbicularis_v_phaeophyscia_rubropulchra',
        'inat_observed_Eastern_Ribbonsnake_vs_Western_Ribbon_Snake',
        'ml_photo_rating_12_vs_45_v1',
        'cct_coyote_vs_bobcat_loc_120_night',
        'inat_unobserved_otiorhynchus_ovatus_v_otiorhynchus_singularis',
        'nabirds_species_classification_louwat_norwat',
        'nabirds_species_classification_houwre_winwre3',
        'nabirds_species_classification_dowwoo_haiwoo',
        'inat_unobserved_tillandsia_balbisiana_v_tillandsia_bartramii',
        'nabirds_species_classification_gresca_lessca',
        'cct_opossum_vs_empty_loc_43_night',
        'ml_tag_feeding_young',
        'inat_observed_Groove-billed_Ani_vs_Smooth-billed_Ani',
        'nabirds_species_classification_solsan_sposan',
        'ml_tag_dead',
        'ml_age_whimbrel',
        'inat_non_species_mating_common_green_darner',
        'ml_age_western_sandpiper',
        'inat_non_species_mating_argia_vivida',
        'nabirds_species_classification_linspa_savspa',
        'nabirds_species_classification_amecro_fiscro',
        'glc_barren_vs_scrub',
        'nabirds_species_classification_rosgoo_snogoo',
        'ml_age_rough_legged_hawk',
        'nabirds_species_classification_commer_rebmer',
        'nabirds_species_classification_cavswa_cliswa',
        'inat_unobserved_polystichum_aculeatum_v_polystichum_setiferum',
        'inat_non_species_dead_common_garter_snake',
        'nabirds_species_classification_herthr_swathr',
        'inat_observed_Yellow-backed_Spiny_Lizard_vs_Desert_Spiny_Lizard',
        'inat_non_species_birds_near_signs',
        'glc_deciduous_forest_vs_evergreen_forest',
        'fgvcx_icassava_healthy_vs_sick',
        'inat_unobserved_panus_conchatus_v_panus_neostrigosus',
        'inat_observed_Blue_Mussel_vs_California_Mussel',
        'nabirds_species_classification_amekes_merlin',
        'inat_observed_Brown_House_Spider_vs_False_Black_Widow'
    ]
    features = [
        "imagenet_supervised",
        "imagenet_moco_v2",
        "imagenet_simclr",
        "imagenet_swav",

        "inat2021_mini_moco_v2",
        "inat2021_mini_simclr",
        "inat2021_mini_swav",
        "inat2021_mini_supervised",

        "inat2021_supervised",
        'inat2021_simclr',
    ]
    for d_name in reversed(data_names):
        _main(feature_name, d_name)


def main_data(data_name):
    features = [
        "imagenet_supervised",
        "imagenet_moco_v2",
        "imagenet_simclr",
        "imagenet_swav",

        "inat2021_mini_moco_v2",
        "inat2021_mini_simclr",
        "inat2021_mini_swav",
        "inat2021_mini_supervised",

        "inat2021_supervised",
        'inat2021_simclr',
    ]
    for f_name in reversed(features):
        _main(f_name, data_name)


if __name__ == '__main__':
    main_feat()
