#ifndef _decisiontrace_actioneval_hpp_INCLUDED
#define _decisiontrace_actioneval_hpp_INCLUDED

#include <cstdint>
#include <cstdio>
#include <unordered_set>

namespace CaDiCaL {

// Mutable state for one solver replay.  Large graph snapshots are never
// retained here: a selected state is inspected, streamed, and released in
// the same decision callback.
struct DecisionTraceRuntime {
  FILE *output = 0;
  bool owns_output = false;
  bool started = false;
  bool finished = false;
  bool stop_requested = false;
  bool stopped_by_horizon = false;

  int64_t callback_attempt_index = 0;
  int64_t raw_decision_index = 0;
  int64_t eligible_state_index = 0;
  int64_t emitted_window_states = 0;
  int emitted_post_restart_state_count = 0;
  int64_t materialized_states = 0;
  int64_t materialization_failures = 0;
  int64_t forced_actions = 0;

  uint64_t prefix_hash = 1469598103934665603ULL;
  std::unordered_set<int64_t> capture_targets;
  std::unordered_set<int64_t> captured_targets;

  bool callback_pending = false;
  bool pending_eligible = false;
  int pending_candidate_count = 0;
  int64_t pending_state_nodes = 0;
  int pending_requested_external_literal = 0;
  int pending_level = 0;
  int64_t pending_trail_size = 0;
  int64_t pending_conflicts = 0;
  int64_t pending_decisions = 0;
  int64_t pending_propagations = 0;
  int64_t pending_restarts = 0;
  uint64_t pending_prefix_hash = 0;

  bool eval_target_reached = false;
  bool eval_force_applied = false;
  bool eval_force_eligible = false;
  bool eval_locator_match = false;
  int eval_native_internal_literal = 0;
  int eval_native_external_literal = 0;
  int eval_applied_internal_literal = 0;
  int eval_applied_external_literal = 0;
  int64_t eval_start_conflicts = 0;
  int64_t eval_start_decisions = 0;
  int64_t eval_start_propagations = 0;
  int64_t eval_start_restarts = 0;
  double eval_timeout_deadline = 0;
  int64_t eval_callback_index = 0;
  int64_t eval_raw_decision_index = 0;
  uint64_t eval_prefix_hash = 0;
};

} // namespace CaDiCaL

#endif
