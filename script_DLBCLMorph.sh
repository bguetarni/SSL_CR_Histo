#!/bin/bash
tmux new-session -d -s 030
for j in {0..3}
do
    tmux new-window
    tmux send 'cd; source .bashrc; conda activate torch_gpu' ENTER
    g=$((j%2))
    tmux send "python /home/isen/bilel/workspace/SSL_CR_Histo/eval_BreastPathQ_SSL.py --num_classes 2 --image_size 112 --train_image_pth /home/isen/bilel/data/SOA/DLBCL-Morph --model_save_pth /home/isen/bilel/workspace/SSL_CR_Histo/checkpoints --name ssl_dlbclmorph_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --model_path /home/isen/bilel/workspace/SSL_CR_Histo/Pretrained_models/BreastPathQ_pretrained_model.pt; 
    python /home/isen/bilel/workspace/SSL_CR_Histo/eval_BreastPathQ_SSL_CR.py --num_classes 2 --image_size 112 --train_image_pth /home/isen/bilel/data/SOA/DLBCL-Morph --model_save_pth /home/isen/bilel/workspace/SSL_CR_Histo/checkpointsCR --name sslcr_dlbclmorph_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --model_path_finetune /home/isen/bilel/workspace/SSL_CR_Histo/checkpoints/ssl_dlbclmorph_$j/best_fine_tuned_model.pt" ENTER
    tmux rename-window dlbclmorph_$j
done
