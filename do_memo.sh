cd  ~/lab/gda/da/TVT
. cmd.sh  0 0 8_9_10_11_12_13_14  --task simclr_encoder_bs512_ep2000_lr0.001_outd64_g3  --dataset DomainNet  --tmux TVT_DomainNet_0

cd  ~/lab/gda/da/TVT
. cmd.sh  1 0 7_8_9_10_11_12_13_14  --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep2000_lr0.001_outd64_g3  --dataset DomainNet  --tmux TVT_DomainNet_1



# ci 0
# cp 1
# cq 2
# cr 3
# cs 4
# ip 5
# iq 6
# ir 7
# is 8
# pq 9
# pr 10
# ps 11
# qr 12
# qs 13
# rs 14
