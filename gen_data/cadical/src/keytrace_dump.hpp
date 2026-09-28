#ifndef _keytrace_dump_hpp_INCLUDED
#define _keytrace_dump_hpp_INCLUDED

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace CaDiCaL {

struct Clause;

enum KeyTraceAssignmentSource {
  KEYTRACE_UNASSIGNED = 0,
  KEYTRACE_DECISION = 1,
  KEYTRACE_PROPAGATION = 2,
  KEYTRACE_ASSERTING_LEARNT = 3,
  KEYTRACE_ASSUMPTION = 4,
  KEYTRACE_EXTERNAL = 5
};

enum KeyTraceEventType { KEYTRACE_EVENT_DECISION = 1 };

struct KeyTraceSnapshot {
  int64_t snapshot_id;
  int64_t raw_decision_index;
  int64_t last_clause_event_id;
  int64_t artifact_replay_event_id;
  int decision_level;
  int64_t conflicts;
  int64_t decisions;
  int64_t propagations;
  int64_t restarts;
  int64_t conflicts_since_restart;
  int64_t decisions_since_restart;
  int64_t propagations_since_restart;
  size_t trail_size;
  size_t num_assigned;
  std::shared_ptr<const std::vector<std::vector<int>>> cnf;
  std::vector<signed char> assignment_value;
  std::vector<int> assignment_level;
  std::vector<signed char> assignment_source;
  std::vector<int> trail_position;
  std::vector<float> activity;
  std::vector<signed char> phase_saved;
};

struct KeyTraceEvent {
  int64_t event_id;
  int type;
  int lit;
  int level;
  int trail_index;
  int64_t snapshot_id;
  signed char assignment_source;
};

struct KeyTracePaperEvent {
  int64_t raw_index;
  char type;
  int lit;
  int level;
};

// Realtime clause lifecycle event used by clause-delta and
// sample-delta-reply modes.  Replay uses artifact ids, while the runtime
// Clause pointer is retained only for diagnostics.
struct KeyTraceClauseDelta {
  int64_t event_id;
  char type;
  int64_t artifact_clause_id;
  int64_t old_artifact_clause_id;
  Clause *clause;
  bool redundant;
  std::vector<int> lits;
  std::vector<int> old_lits;
};

} // namespace CaDiCaL

#endif
