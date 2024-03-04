#!/bin/bash
tmux new-session -d -s 040
for j in {0..3}
do
    tmux new-window
    tmux send 'cd; source .bashrc; conda activate torch_gpu' ENTER
    g=$((j%2 + 2))
    tmux send "python /home/isen/bilel/workspace/SSL_CR_Histo/eval_BreastPathQ_SSL.py --num_classes 4 --image_size 512 --train_image_pth /home/isen/bilel/data/SOA/BCI --model_save_pth /home/isen/bilel/workspace/SSL_CR_Histo/checkpoints --name ssl_bci_$j --gpu $g --dataset bci --model_path /home/isen/bilel/workspace/SSL_CR_Histo/Pretrained_models/BreastPathQ_pretrained_model.pt; 
    python /home/isen/bilel/workspace/SSL_CR_Histo/eval_BreastPathQ_SSL_CR.py --num_classes 4 --image_size 512 --train_image_pth /home/isen/bilel/data/SOA/BCI --model_save_pth /home/isen/bilel/workspace/SSL_CR_Histo/checkpointsCR --name sslcr_bci_$j --gpu $g --dataset bci --model_path_finetune /home/isen/bilel/workspace/SSL_CR_Histo/checkpoints/ssl_bci_$j/best_fine_tuned_model.pt" ENTER
    tmux rename-window bci_$j
done
