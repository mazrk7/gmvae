WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=eval \
    --model=gmvae \
    --latent_size=128 \
    --hidden_size=512 \
    --logdir="$WS_DIR/checkpoints" \
    --num_samples=200 \
    --num_generations=10 \
    --batch_size=64 \
    --split=test