#!/bin/bash
# Example usage: ./multiple_feature_extraction.sh 0 5 shift

GPU_ID=$1
N_RUNS=$2
CONFIG=$3

for ((i=1; i<=N_RUNS; i++))
do
    echo "Run $i / $N_RUNS | GPU $GPU_ID | CONFIG $CONFIG"

    case $CONFIG in

    shift)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_shift \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --spatial_shift
        ;;

    jitter)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_jitter \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --color_jitter
        ;;

    gaussian)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_gaussian \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --gaussian_smooth --gaussian_sigma_min 0.4 --gaussian_sigma_max 0.9 --gaussian_kernel_size 3
        ;;

    zoom)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_zoom \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --zoom
        ;;

    jitter_shift)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_jitter_shift \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --color_jitter --spatial_shift
        ;;

    rotation_zoom)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_rotation_zoom \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --zoom --rotation
        ;;

    shift_hflip)
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_feature.py \
        --model_type DFER --emb_red spatial \
        --path_dataset partA/video/video_frontalized_interpolated_resolution_original \
        --path_labels partA/starting_point/samples.csv \
        --saving_folder_path partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol_shift_hflip \
        --backbone_type video --gp --save_as_safetensors \
        --stride_window 16 --stride_inside_window 1 --float_16 --save_big_feature \
        --spatial_shift --h_flip
        ;;

    *)
        echo "Invalid config"
        exit 1
        ;;

    esac

done