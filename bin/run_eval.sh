WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=eval \
    --model=gmvae \
    --latent_size=64 \
    --hidden_size=256 \
    --num_layers=2 \
    --logdir="$WS_DIR/checkpoints" \
    --num_samples=1000 \
    --batch_size=16 \
    --split=test