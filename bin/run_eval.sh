WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=eval \
    --model=vae_gmp \
    --latent_size=10 \
    --hidden_size=64 \
    --logdir="$WS_DIR/checkpoints" \
    --num_samples=2000 \
    --batch_size=16 \
    --split=test