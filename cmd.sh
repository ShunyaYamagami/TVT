function process_args {
    declare -A args
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3

    # 残りの名前付き引数を解析
    local dataset="Office31"
    if [ $dataset = 'Office31' ]; then
        local task=(
            # "original_uda"
            # "true_domains"
            # "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            # "simclr_bs512_ep300_g3_shfl"
            # "simple_bs512_ep300_g3_AE_outd64_shfl"
            "contrastive_rpl_dim512_wght0.6_AE_bs256_ep300_outd64_g3"
        )
    elif [ $dataset = 'OfficeHome' ]; then
        local task=(
            # "original_uda"
            "true_domains"
            "simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep1000_g3_shfl"
        )
    fi

    echo "gpu_i: $gpu_i"
    echo "exec_num: $exec_num"
    echo "dset_num: $dset_num"
    echo -e ''

    ##### データセット設定
    if [ $dataset = 'Office31' ]; then
        dsetlist=("amazon_dslr" "webcam_amazon" "dslr_webcam")
    elif [ $dataset = 'OfficeHome' ]; then
        dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
    elif [ $dataset = 'DomainNet' ]; then
        dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
    else
        echo "不明なデータセット: $dataset" >&2
        return 1
    fi

    COMMAND=''
    COMMAND+="conda deactivate && conda deactivate"
    COMMAND+=" && conda activate tvt"

    for tsk in "${task[@]}"; do
        if [ $dset_num -eq -1 ]; then
            for dset in "${dsetlist[@]}"; do
                COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  python3 main.py \
                    --train_batch_size 20 \
                    --dataset $dataset \
                    --dset $dset \
                    --task $tsk \
                    --model_type ViT-B_16 \
                    --pretrained_dir checkpoint/imagenet21k_ViT-B_16.npz \
                    --num_steps 5000 \
                    --beta 0.1 \
                    --gamma 0.01 \
                    --use_im \
                    --theta 0.1 \
                    --img_size 256
                "
            done
        else
            dset=${dsetlist[$dset_num]}
            COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  python3 main.py \
                --train_batch_size 20 \
                --dataset $dataset \
                --dset $dset \
                --task $tsk \
                --model_type ViT-B_16 \
                --pretrained_dir checkpoint/imagenet21k_ViT-B_16.npz \
                --num_steps 5000 \
                --beta 0.1 \
                --gamma 0.01 \
                --use_im \
                --theta 0.1 \
                --img_size 256
            "
        fi
    done

    ###### 実行. 
    echo $COMMAND
    echo ''
    eval $COMMAND
}

# 最初の3つの引数をチェック
if [ "$#" -lt 1 ]; then
    echo "エラー: 引数が足りません。最初の1つの引数は必須です。" >&2
    return 1
fi
########## Main ##########
process_args "$@"