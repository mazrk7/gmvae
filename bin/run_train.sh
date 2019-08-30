WS_DIR=$HOME/workspace/gmvae

python $WS_DIR/scripts/run_gmvae.py \
    --mode=train \
    --model=gmvae \
    --latent_size=128 \
    --hidden_size=512 \
    --num_layers=3 \
    --logdir="$WS_DIR/checkpoints" \
    --summarise_every=1000 \
    --early_stop_rounds=10000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --gpu_id="0" \
    --gpu_num="0"