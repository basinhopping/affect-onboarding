Night 12 (2026-10-11, Sunday night Pacific), prepared in the Cowork container: the pooling test on the two finalist
CNNs, for Dr. Luitel's question about pooling and SD pooling (meeting of 2026-10-12). Harrison asked for a run tonight
("We can do anotyher run tonight though") and chose this one from the two offered.

make_night12.py, make_night12.log   writes sweeps/cnn_3block_pooling_1.json and sweeps/cnn_final_3_pooling_1.json from
                        the selection records threeblock_final and cnn_final_3, after checking with the repository's
                        sweep expansion that each holds 100 jobs, that each base hashes to its record's configuration
                        (e0c211fcced3, 7977141b6ba3) and that each variant differs from its base in model.pooling only.
                        tests/test_sweep.py with the two files: 20 tests, OK. Tree digest with them (the recipe of
                        HANDOFF.md, the same on the PC and in the container):
                        5a49cdcee1eca39c9d1d64b3cc079f45638506bfc1a3507b2a38d72870a31b5b
RULE_night12_tests.txt  which runs are made, which test files are read and which comparisons are made (families N12A,
                        N12B), fixed at 06:27 UTC on 2026-10-12, before any file of night 12 left the node, and amended
                        at 06:44 UTC, still before: 2,000 draws instead of 20,000 (the dry run's timing).
                        SHA-256 as fixed c67dad632295d91c8a4fd8bc78b92ed2358098c753d97e5e960c670d5dc76ea9;
                        as amended 57ec09965e0b3f7619641c29967bf3170a2e46256aaa4ccbd0cd3b387bc4ee67
lines/                  the night's lines, each checked for bash syntax:
  1_push.txt            the PC: push.ps1 (one password prompt). The first push with the GitHub version of push.ps1,
                        which reads the node's login from node.local.txt.
  2_unpack_check.txt    the node: unpack.sh installs the newest archive; expects TREE OK 5a49cdce..., then check.sh:
                        "Ran 202 tests", "OK", "check: all three passed", "check.sh exit 0".
  3_night12_start.txt   the node: refuses if a gpu or night12 session exists or the queue holds a job; prints the GPUs'
                        use; then, in tmux session night12, smoke-tests and enqueues the two sweeps on GPU 0
                        (logs/enqueue_night12.txt), requeues stale running jobs, removes the stop files and runs
                        gpu_schedule.sh with two steps: night now (3 workers on every GPU no other user is on) and end
                        at 08:00 Pacific on Monday 2026-10-12 (STOP). The workers also stop when the queue is empty.
                        Run on simulated nodes (sim_start/, adapted from night 11's): both sweeps smoke-tested on GPU 0
                        and enqueued, GPUs 0 to 2 started and GPU 3 (another user's process) left alone, STOP at the
                        end step; with a job pending, or with a gpu session, the line refuses and starts nothing.
  4_night12_status.txt  any time later: the enqueue log's end, the sessions, the schedule log, each sweep's jobs by state.
  5_pack_night12.txt    Monday morning, once the queue is empty or the end step has run: refuses while workers run;
                        closes the night12 session; counts each sweep's jobs; copies the logs written since 2026-10-11
                        22:00 UTC into artifacts/logs_night12; packs the two sweeps' runs and records, those logs, and
                        the smoke runs and checks made since then. Then pull.ps1 with the name pack.sh prints.
