kaw324@COECIS-L4XVTJ99KM adscheduler % python3 scripts/run_schedule_benchmark.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd \
  --schedule ror_remat --schedule for_remat --schedule jacrev_remat --schedule jacfwd_remat \
  --outer-steps 500 \
  --eval-every 10 \
  --target-accuracy 0.85 \
  --meta-batch-size 4 \
  --meta-test-tasks 64

=== Benchmark Summary ===
schedules: ror, for, jacrev, jacfwd, ror_remat, for_remat, jacrev_remat, jacfwd_remat
outer_steps: 500 eval_every: 10
target_meta_test_accuracy: 0.85

[ror]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 0.088
  p50_outer_step_time_ms: 0.087
  p90_outer_step_time_ms: 0.090
  compile_overhead_ms: 179.127
  peak_host_memory_mb: 339.12
  peak_device_memory_mb: n/a
  ir_total_equations: 135
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 118

[for]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 2.073
  p50_outer_step_time_ms: 2.001
  p90_outer_step_time_ms: 2.214
  compile_overhead_ms: 270.913
  peak_host_memory_mb: 355.53
  peak_device_memory_mb: n/a
  ir_total_equations: 470
  ir_max_loop_nesting: 0
  ir_num_higher_order_sites: 407

[jacrev]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 0.131
  p50_outer_step_time_ms: 0.128
  p90_outer_step_time_ms: 0.131
  compile_overhead_ms: 253.568
  peak_host_memory_mb: 360.80
  peak_device_memory_mb: n/a
  ir_total_equations: 215
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 161

[jacfwd]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 1.673
  p50_outer_step_time_ms: 1.645
  p90_outer_step_time_ms: 1.753
  compile_overhead_ms: 188.072
  peak_host_memory_mb: 366.72
  peak_device_memory_mb: n/a
  ir_total_equations: 213
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 160

[ror_remat]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 0.086
  p50_outer_step_time_ms: 0.084
  p90_outer_step_time_ms: 0.087
  compile_overhead_ms: 172.325
  peak_host_memory_mb: 367.19
  peak_device_memory_mb: n/a
  ir_total_equations: 152
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 134

[for_remat]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 2.009
  p50_outer_step_time_ms: 1.980
  p90_outer_step_time_ms: 2.059
  compile_overhead_ms: 255.708
  peak_host_memory_mb: 368.47
  peak_device_memory_mb: n/a
  ir_total_equations: 476
  ir_max_loop_nesting: 0
  ir_num_higher_order_sites: 413

[jacrev_remat]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 0.129
  p50_outer_step_time_ms: 0.125
  p90_outer_step_time_ms: 0.130
  compile_overhead_ms: 230.888
  peak_host_memory_mb: 368.47
  peak_device_memory_mb: n/a
  ir_total_equations: 233
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 177

[jacfwd_remat]
  final_meta_test_accuracy: 0.8223
  best_meta_test_accuracy: 0.8232
  outer_iterations_to_target: not reached
  final_meta_train_loss: 0.393952
  avg_outer_step_time_ms: 1.661
  p50_outer_step_time_ms: 1.646
  p90_outer_step_time_ms: 1.715
  compile_overhead_ms: 192.889
  peak_host_memory_mb: 368.47
  peak_device_memory_mb: n/a
  ir_total_equations: 215
  ir_max_loop_nesting: 1
  ir_num_higher_order_sites: 160

[Summary]
Current results already suggest that schedules can differ a lot in cost.
For example, ror and jacrev are very fast. for and jacfwd are much slower. remat hurts here.