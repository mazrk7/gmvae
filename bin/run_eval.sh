WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=eval \
    --model=gmvae \
    --latent_size=20 \
    --hidden_size=128 \
    --logdir="$WS_DIR/checkpoints" \
    --num_samples=2000 \
    --batch_size=16 \
    --split=test