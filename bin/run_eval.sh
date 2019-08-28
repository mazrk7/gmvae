WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=eval \
    --model=gmvae \
    --latent_size=128 \
    --hidden_size=512 \
    --num_layers=2 \
    --logdir="$WS_DIR/checkpoints" \
    --num_samples=100 \
    --num_generations=10 \
    --batch_size=16 \
    --split=test