. cmd.sh  0 0 0_1_2  --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep3000_outd64_g3  --dataset OfficeHome  --tmux SDAT_2
. cmd.sh  1 0 1  --task simclr_bs512_ep1000_g3_shfl  --dataset OfficeHome  --tmux TVT_autoencoder
. cmd.sh  1 0 0_2_3_4_5  --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep3000_outd64_g3  --dataset OfficeHome  --tmux TVT_autoencoder
