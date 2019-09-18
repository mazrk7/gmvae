WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=train \
    --model=gmvae \
    --latent_size=128 \
    --hidden_size=512 \
    --logdir="$WS_DIR/checkpoints" \
    --summarise_every=1000 \
    --early_stop_rounds=5000 \
    --batch_size=64 \
    --learning_rate=0.001 \
    --gpu_id="0" \
    --gpu_num="0"