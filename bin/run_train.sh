WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=train \
    --model=vae \
    --latent_size=10 \
    --hidden_size=64 \
    --logdir="$WS_DIR/checkpoints" \
    --summarise_every=1000 \
    --batch_size=16 \
    --learning_rate=0.001 \
    --gpu_id="0" \
    --gpu_num="0"