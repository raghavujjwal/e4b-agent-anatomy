#!/bin/bash
# Tier-2 replay in parallel shards (S2.3). Run inside WSL.
#   bash train/s2_run_replay.sh <dir with jobs.jsonl> <number of shards>
#   FETCH_TIMEOUT=400 RETRY_FETCH=1 bash train/s2_run_replay.sh <dir> 3      # second pass: only the repositories whose download failed
# Writes <dir>/replayed.shard<i>.jsonl and, when all shards are done, merges them into <dir>/replayed.jsonl.
# Safe to stop and start again (also with another number of shards): a started shard skips what any shard file in <dir> has settled and appends.
# The time goes to downloading the repository tarballs from GitHub (a few hundred KB/s per connection, and the link is shared), not to the CPU.
# A big repository would hold a shard for minutes, so the first pass gives up on a download after 75 s and the second pass takes those slowly.
DIR=${1:?directory with jobs.jsonl}
N=${2:-4}
T=${FETCH_TIMEOUT:-75}
EXTRA=""
[ -n "$RETRY_FETCH" ] && EXTRA="--retry-fetch"
cd "$(dirname "$0")/.." || exit 1
# The jobs file (about 50 MB) is read once over the slow /mnt/d mount; with every shard reading it there, the start took minutes.
mkdir -p /dev/shm/s2
JOBS=/dev/shm/s2/jobs.$(basename "$DIR").jsonl
cp "$DIR/jobs.jsonl" "$JOBS" || exit 1
for i in $(seq 0 $((N - 1))); do
  python3 -u train/s2_replay.py --jobs "$JOBS" --out "$DIR/replayed.shard$i.jsonl" --resume $EXTRA --fetch-timeout "$T" \
    --cache /mnt/d/data/s2/tarballs --shard "$i/$N" --tmp "/dev/shm/s2/shard$i" >> "$DIR/replay.shard$i.log" 2>&1 &
done
wait
python3 train/s2_replay.py --merge "$DIR"
rm -f "$JOBS"
