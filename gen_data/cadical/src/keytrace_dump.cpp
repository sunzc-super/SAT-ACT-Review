#include "internal.hpp"

#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <iomanip>
#include <map>
#include <functional>
#include <sstream>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>

namespace CaDiCaL {

static const char *source_name (signed char source) {
  switch (source) {
  case KEYTRACE_DECISION:
    return "DECISION";
  case KEYTRACE_PROPAGATION:
    return "PROPAGATION";
  case KEYTRACE_ASSERTING_LEARNT:
    return "ASSERTING_LEARNT";
  case KEYTRACE_ASSUMPTION:
    return "ASSUMPTION";
  case KEYTRACE_EXTERNAL:
    return "EXTERNAL";
  default:
    return "UNASSIGNED";
  }
}

static std::string trim_trailing_slashes (const std::string &path) {
  std::string res = path;
  while (res.size () > 1 && res[res.size () - 1] == '/')
    res.resize (res.size () - 1);
  return res;
}

static std::string join_path (const std::string &dir,
                              const std::string &name) {
  if (dir.empty () || dir == ".")
    return name;
  if (dir[dir.size () - 1] == '/')
    return dir + name;
  return dir + "/" + name;
}

static bool dir_exists (const std::string &path) {
  struct stat st;
  return !stat (path.c_str (), &st) && S_ISDIR (st.st_mode);
}

static bool path_exists (const std::string &path) {
  struct stat st;
  return !stat (path.c_str (), &st);
}

static bool mkdir_p (const std::string &path) {
  if (path.empty ())
    return false;
  if (dir_exists (path))
    return true;
  if (path_exists (path))
    return false;
  size_t pos = 0;
  while (true) {
    pos = path.find ('/', pos + 1);
    const std::string part =
        pos == std::string::npos ? path : path.substr (0, pos);
    if (!part.empty () && !dir_exists (part)) {
      if (mkdir (part.c_str (), 0777) && errno != EEXIST)
        return false;
      if (!dir_exists (part))
        return false;
    }
    if (pos == std::string::npos)
      break;
  }
  return dir_exists (path);
}

static std::string keytrace_dump_dir (const Internal *internal) {
  return trim_trailing_slashes (internal->keytrace_dump_path);
}

static std::string artifacts_dir (const Internal *internal) {
  return join_path (keytrace_dump_dir (internal), "artifacts");
}

static bool valid_keytrace_output_prefix_char (char ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
}

static bool valid_keytrace_output_prefix (const char *prefix) {
  if (!prefix || !*prefix)
    return false;
  if (!strcmp (prefix, ".") || !strcmp (prefix, ".."))
    return false;
  for (const char *p = prefix; *p; p++)
    if (!valid_keytrace_output_prefix_char (*p))
      return false;
  return true;
}

static std::string prefixed_keytrace_file_name (const Internal *internal,
                                                const std::string &suffix) {
  return internal->keytrace_output_prefix + "." + suffix;
}

static std::string artifact_path (const Internal *internal,
                                  const std::string &suffix) {
  return join_path (artifacts_dir (internal),
                    prefixed_keytrace_file_name (internal, suffix));
}

static std::string default_work_dir () {
  const char *tmp = getenv ("TMPDIR");
  if (tmp && *tmp)
    return tmp;
  return "/tmp";
}

static bool put (File *file, const std::string &s) {
  return file->put (s.c_str ());
}

static void put_int_array (File *file, const std::vector<int> &values,
                           size_t begin) {
  file->put ('[');
  for (size_t i = begin; i < values.size (); i++) {
    if (i != begin)
      file->put (',');
    file->put ((int64_t) values[i]);
  }
  file->put (']');
}

static void put_char_array (File *file, const std::vector<signed char> &values,
                            size_t begin) {
  file->put ('[');
  for (size_t i = begin; i < values.size (); i++) {
    if (i != begin)
      file->put (',');
    file->put ((int64_t) values[i]);
  }
  file->put (']');
}

static void put_float_array (File *file, const std::vector<float> &values,
                             size_t begin) {
  file->put ('[');
  for (size_t i = begin; i < values.size (); i++) {
    if (i != begin)
      file->put (',');
    char buffer[64];
    double value = (double) values[i];
    if (!std::isfinite (value)) {
      if (std::isnan (value))
        value = 0.0;
      else
        value = value < 0 ? -std::numeric_limits<float>::max () :
                            std::numeric_limits<float>::max ();
    }
    snprintf (buffer, sizeof buffer, "%.9g", value);
    file->put (buffer);
  }
  file->put (']');
}

void Internal::enable_keytrace_dump (const char *path) {
  keytrace_dump = true;
  keytrace_dump_path = path;
  if (keytrace_work_dir.empty ())
    keytrace_work_dir = default_work_dir ();
}

void Internal::set_keytrace_dump_profile (const char *profile) {
  if (strcmp (profile, "jsonl"))
    error ("invalid keytrace dump profile '%s'", profile);
}

void Internal::set_keytrace_dump_debug (bool value) {
  keytrace_dump_debug = value;
}

void Internal::set_keytrace_dump_learnts (bool value) {
  keytrace_dump_learnts = value;
}

void Internal::set_keytrace_dump_paper_original (bool) {
  keytrace_dump_paper_original = true;
}

void Internal::set_keytrace_save_artifacts (bool value) {
  keytrace_save_artifacts = value;
}

void Internal::set_keytrace_work_dir (const char *path) {
  keytrace_work_dir = path && *path ? path : default_work_dir ();
}

void Internal::set_keytrace_output_prefix (const char *prefix) {
  if (!valid_keytrace_output_prefix (prefix))
    error ("invalid keytrace output prefix '%s'", prefix ? prefix : "");
  keytrace_output_prefix = prefix;
}

void Internal::set_keytrace_search_inprocessing (int value) {
  if (value < -1 || value > 1)
    error ("invalid keytrace search inprocessing mode '%d'", value);
  // Kept for CLI/API compatibility.  KeyTrace v1.6 records the active
  // solver CNF lifecycle and does not use this option to change search.
}

void Internal::set_keytrace_sample_top_k (int value) {
  keytrace_sample_top_k = value;
}

void Internal::set_keytrace_profile_mode (const char *mode) {
  if (strcmp (mode, "strict") && strcmp (mode, "trace"))
    error ("invalid keytrace profile mode '%s'", mode);
  keytrace_profile_mode = mode;
}

void Internal::set_keytrace_cnf_mode (const char *mode) {
  if (!strcmp (mode, "sample-first") || !strcmp (mode, "clause-delta") ||
      !strcmp (mode, "sample-delta-reply")) {
    keytrace_cnf_mode = mode;
    return;
  }
  error ("invalid keytrace CNF mode '%s'", mode);
}

void Internal::keytrace_clear () {
  keytrace_event_id = 0;
  keytrace_snapshot_id = 0;
  keytrace_raw_decision_index = 0;
  keytrace_paper_raw_event_index = 0;
  keytrace_last_clause_event_id = 0;
  keytrace_pending_snapshot_id = -1;
  keytrace_conflicts_at_last_restart = stats.conflicts;
  keytrace_decisions_at_last_restart = stats.decisions;
  keytrace_propagations_at_last_restart = stats.propagations.search;
  keytrace_pending_assignment_source = KEYTRACE_PROPAGATION;
  keytrace_search_started = false;
  keytrace_inside_restart = false;
  keytrace_stack.clear ();
  keytrace_trace_prefix_before_final_unsat.clear ();
  keytrace_trace_prefix_before_final_unsat_valid = false;
  keytrace_paper_stack.clear ();
  keytrace_assignment_source_by_var.clear ();
  keytrace_artifact_clause_id_counter = 0;
  keytrace_active_clauses.clear ();
  keytrace_artifact_clause_ids.clear ();
  keytrace_old_artifact_clause_ids.clear ();
  keytrace_base_clauses.clear ();
  keytrace_cached_cnf.reset ();
  keytrace_cnf_dirty = true;
  keytrace_clause_deltas.clear ();
  keytrace_snapshots.clear ();
}

static const char *keytrace_inprocessing_mode (const Internal *internal) {
  if (internal->keytrace_search_inprocessing < 0)
    return "auto";
  if (internal->keytrace_search_inprocessing > 0)
    return "enabled";
  return "disabled";
}

static const char *keytrace_run_result (int res) {
  if (res == 10)
    return "SAT";
  if (res == 20)
    return "UNSAT";
  return "UNKNOWN";
}

static bool keytrace_trace_mode (const Internal *internal) {
  return internal->keytrace_profile_mode == "trace";
}

static bool keytrace_clause_delta_mode (const Internal *internal) {
  return internal->keytrace_cnf_mode == "clause-delta" ||
         internal->keytrace_cnf_mode == "sample-delta-reply";
}

static bool keytrace_direct_snapshot_mode (const Internal *internal) {
  return internal->keytrace_cnf_mode != "clause-delta";
}

static std::vector<int> clause_lits (Clause *c) {
  std::vector<int> lits;
  for (const auto &lit : *c)
    lits.push_back (lit);
  std::sort (lits.begin (), lits.end ());
  return lits;
}

static std::vector<std::pair<std::vector<int>, Clause *>> collect_current_clause_entries (
    Internal *internal) {
  std::vector<std::pair<std::vector<int>, Clause *>> entries;
  for (auto c : internal->clauses) {
    if (!c || c->garbage)
      continue;
    entries.push_back (std::make_pair (clause_lits (c), c));
  }
  std::sort (entries.begin (), entries.end (),
             [] (const std::pair<std::vector<int>, Clause *> &a,
                 const std::pair<std::vector<int>, Clause *> &b) {
               if (a.first != b.first)
                 return a.first < b.first;
               return std::less<Clause *>() (a.second, b.second);
             });
  return entries;
}

static std::vector<std::vector<int>> collect_current_cnf (
    Internal *internal) {
  std::vector<std::vector<int>> cnf;
  const auto entries = collect_current_clause_entries (internal);
  for (const auto &entry : entries)
    cnf.push_back (entry.first);
  return cnf;
}

static std::shared_ptr<const std::vector<std::vector<int>>> current_cnf_snapshot (
    Internal *internal) {
  if (!internal->keytrace_cached_cnf || internal->keytrace_cnf_dirty) {
    internal->keytrace_cached_cnf =
        std::make_shared<const std::vector<std::vector<int>>> (
            collect_current_cnf (internal));
    internal->keytrace_cnf_dirty = false;
  }
  return internal->keytrace_cached_cnf;
}

static void append_realtime_clause_delta (Internal *internal, char type,
                                          int64_t artifact_clause_id,
                                          int64_t old_artifact_clause_id,
                                          Clause *clause,
                                          const std::vector<int> &lits,
                                          const std::vector<int> &old_lits) {
  KeyTraceClauseDelta delta;
  delta.event_id = ++internal->keytrace_last_clause_event_id;
  delta.type = type;
  delta.artifact_clause_id = artifact_clause_id;
  delta.old_artifact_clause_id = old_artifact_clause_id;
  delta.clause = clause;
  delta.redundant = clause ? clause->redundant : false;
  delta.lits = lits;
  delta.old_lits = old_lits;
  internal->keytrace_clause_deltas.push_back (delta);
}

static void note_direct_cnf_lifecycle_event (Internal *internal) {
  if (keytrace_direct_snapshot_mode (internal) &&
      !keytrace_clause_delta_mode (internal))
    ++internal->keytrace_last_clause_event_id;
}

static void require_tracked_realtime_clause (Internal *internal, Clause *c,
                                             const char *op) {
  if (internal->keytrace_active_clauses.find (c) ==
          internal->keytrace_active_clauses.end () ||
      internal->keytrace_artifact_clause_ids.find (c) ==
          internal->keytrace_artifact_clause_ids.end ())
    internal->error ("keytrace clause-delta %s of untracked clause", op);
}

static void keytrace_realtime_replace_clause (Internal *internal, Clause *c,
                                              const char *op) {
  require_tracked_realtime_clause (internal, c, op);
  const auto active_pos = internal->keytrace_active_clauses.find (c);
  const auto id_pos = internal->keytrace_artifact_clause_ids.find (c);
  const std::vector<int> old_lits = active_pos->second;
  const std::vector<int> new_lits = clause_lits (c);
  if (old_lits == new_lits)
    return;
  const int64_t old_id = id_pos->second;
  const int64_t new_id = ++internal->keytrace_artifact_clause_id_counter;
  append_realtime_clause_delta (internal, 'r', new_id, old_id, c, new_lits,
                                old_lits);
  active_pos->second = new_lits;
  id_pos->second = new_id;
  internal->keytrace_old_artifact_clause_ids[c] = new_id;
}

void Internal::keytrace_after_new_clause (Clause *c) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (keytrace_direct_snapshot_mode (this))
    keytrace_cnf_dirty = true;
  if (!keytrace_clause_delta_mode (this) || !c || c->garbage) {
    if (c && !c->garbage)
      note_direct_cnf_lifecycle_event (this);
    return;
  }
  const std::vector<int> lits = clause_lits (c);
  const int64_t artifact_id = ++keytrace_artifact_clause_id_counter;
  keytrace_active_clauses[c] = lits;
  keytrace_artifact_clause_ids[c] = artifact_id;
  keytrace_old_artifact_clause_ids[c] = artifact_id;
  append_realtime_clause_delta (this, 'a', artifact_id, 0, c, lits,
                                std::vector<int> ());
}

void Internal::keytrace_clause_id_changed (Clause *, int64_t) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (keytrace_direct_snapshot_mode (this))
    keytrace_cnf_dirty = true;
}

void Internal::keytrace_clause_relocated (Clause *from, Clause *to) {
  if (!keytrace_dump || !keytrace_search_started ||
      !keytrace_clause_delta_mode (this) || !from || !to)
    return;
  const auto active_pos = keytrace_active_clauses.find (from);
  if (active_pos == keytrace_active_clauses.end ())
    return;
  keytrace_active_clauses[to] = active_pos->second;
  keytrace_active_clauses.erase (active_pos);
  const auto id_pos = keytrace_artifact_clause_ids.find (from);
  if (id_pos != keytrace_artifact_clause_ids.end ()) {
    keytrace_artifact_clause_ids[to] = id_pos->second;
    keytrace_artifact_clause_ids.erase (id_pos);
  }
  const auto old_id_pos = keytrace_old_artifact_clause_ids.find (from);
  if (old_id_pos != keytrace_old_artifact_clause_ids.end ()) {
    keytrace_old_artifact_clause_ids[to] = old_id_pos->second;
    keytrace_old_artifact_clause_ids.erase (old_id_pos);
  }
}

void Internal::keytrace_clause_modified (Clause *c) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (keytrace_direct_snapshot_mode (this))
    keytrace_cnf_dirty = true;
  if (keytrace_clause_delta_mode (this) && c && !c->garbage)
    keytrace_realtime_replace_clause (this, c, "replace");
  else if (c && !c->garbage)
    note_direct_cnf_lifecycle_event (this);
}

void Internal::keytrace_clause_deleted (Clause *c) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (keytrace_direct_snapshot_mode (this))
    keytrace_cnf_dirty = true;
  if (!keytrace_clause_delta_mode (this) || !c) {
    if (c && !c->garbage)
      note_direct_cnf_lifecycle_event (this);
    return;
  }
  const auto active_pos = keytrace_active_clauses.find (c);
  const auto id_pos = keytrace_artifact_clause_ids.find (c);
  if (active_pos == keytrace_active_clauses.end () ||
      id_pos == keytrace_artifact_clause_ids.end ()) {
    if (!c->garbage)
      error ("keytrace clause-delta delete of untracked clause");
    return;
  }
  append_realtime_clause_delta (this, 'd', id_pos->second, 0, c,
                                active_pos->second, std::vector<int> ());
  keytrace_active_clauses.erase (active_pos);
  keytrace_artifact_clause_ids.erase (id_pos);
  keytrace_old_artifact_clause_ids.erase (c);
}

void Internal::keytrace_clause_shrunk (Clause *c,
                                       const std::vector<int> &) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (keytrace_direct_snapshot_mode (this))
    keytrace_cnf_dirty = true;
  if (keytrace_clause_delta_mode (this) && c && !c->garbage)
    keytrace_realtime_replace_clause (this, c, "shrink");
  else if (c && !c->garbage)
    note_direct_cnf_lifecycle_event (this);
}

void Internal::keytrace_begin_search () {
  if (!keytrace_dump || keytrace_search_started)
    return;
  keytrace_clear ();
  keytrace_search_started = true;
  if ((int) keytrace_assignment_source_by_var.size () <= max_var)
    keytrace_assignment_source_by_var.resize (max_var + 1, KEYTRACE_UNASSIGNED);
  for (auto lit : trail) {
    const int idx = vidx (lit);
    if (idx <= max_var)
      keytrace_assignment_source_by_var[idx] = KEYTRACE_PROPAGATION;
  }
  keytrace_base_clauses.clear ();
  const auto base_entries = collect_current_clause_entries (this);
  for (const auto &entry : base_entries)
    keytrace_base_clauses.push_back (entry.first);
  keytrace_cached_cnf =
      std::make_shared<const std::vector<std::vector<int>>> (
          keytrace_base_clauses);
  keytrace_cnf_dirty = false;
  if (keytrace_clause_delta_mode (this)) {
    keytrace_artifact_clause_id_counter = 0;
    keytrace_last_clause_event_id = 0;
    keytrace_active_clauses.clear ();
    keytrace_artifact_clause_ids.clear ();
    keytrace_old_artifact_clause_ids.clear ();
    keytrace_clause_deltas.clear ();
    for (const auto &entry : base_entries) {
      Clause *c = entry.second;
      const int64_t artifact_id = ++keytrace_artifact_clause_id_counter;
      keytrace_active_clauses[c] = entry.first;
      keytrace_artifact_clause_ids[c] = artifact_id;
      keytrace_old_artifact_clause_ids[c] = artifact_id;
    }
  }
}

void Internal::keytrace_before_search_assign (signed char source) {
  if (keytrace_dump)
    keytrace_pending_assignment_source = source;
}

void Internal::keytrace_assignment (int lit, signed char source) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  const int idx = vidx (lit);
  if ((int) keytrace_assignment_source_by_var.size () <= idx)
    keytrace_assignment_source_by_var.resize (idx + 1, KEYTRACE_UNASSIGNED);
  keytrace_assignment_source_by_var[idx] = source;
  if (source == KEYTRACE_UNASSIGNED)
    return;
  if (source != KEYTRACE_DECISION)
    return;

  KeyTraceEvent event;
  event.event_id = ++keytrace_event_id;
  event.type = KEYTRACE_EVENT_DECISION;
  event.lit = lit;
  event.level = var (idx).level;
  event.trail_index = var (idx).trail;
  event.snapshot_id = keytrace_pending_snapshot_id;
  event.assignment_source = source;
  keytrace_stack.push_back (event);

  KeyTracePaperEvent paper;
  paper.raw_index = ++keytrace_paper_raw_event_index;
  paper.type = 'D';
  paper.lit = lit;
  paper.level = event.level;
  keytrace_paper_stack.push_back (paper);
}

void Internal::keytrace_before_decision () {
  if (!keytrace_dump || !keytrace_search_started)
    return;

  const int64_t raw_decision_index = ++keytrace_raw_decision_index;

  KeyTraceSnapshot snapshot;
  snapshot.snapshot_id = ++keytrace_snapshot_id;
  snapshot.raw_decision_index = raw_decision_index;
  snapshot.last_clause_event_id = keytrace_last_clause_event_id;
  snapshot.artifact_replay_event_id = keytrace_last_clause_event_id;
  snapshot.decision_level = level;
  snapshot.trail_size = trail.size ();
  snapshot.num_assigned = num_assigned;
  snapshot.conflicts = stats.conflicts;
  snapshot.decisions = stats.decisions;
  snapshot.propagations = stats.propagations.search;
  snapshot.restarts = stats.restarts;
  snapshot.conflicts_since_restart = stats.conflicts - keytrace_conflicts_at_last_restart;
  snapshot.decisions_since_restart = stats.decisions - keytrace_decisions_at_last_restart;
  snapshot.propagations_since_restart = stats.propagations.search - keytrace_propagations_at_last_restart;

  if (keytrace_direct_snapshot_mode (this))
    snapshot.cnf = current_cnf_snapshot (this);
  snapshot.assignment_value.resize (max_var + 1, 0);
  snapshot.assignment_level.resize (max_var + 1, -1);
  snapshot.assignment_source.resize (max_var + 1, KEYTRACE_UNASSIGNED);
  snapshot.trail_position.resize (max_var + 1, -1);
  snapshot.activity.resize (max_var + 1, 0);
  snapshot.phase_saved.resize (max_var + 1, 0);
  for (int idx = 1; idx <= max_var; idx++) {
    snapshot.assignment_value[idx] = val (idx);
    snapshot.assignment_level[idx] = val (idx) ? var (idx).level : -1;
    snapshot.trail_position[idx] = val (idx) ? var (idx).trail : -1;
    if (idx < (int) keytrace_assignment_source_by_var.size ())
      snapshot.assignment_source[idx] = keytrace_assignment_source_by_var[idx];
    snapshot.activity[idx] = (float) score (idx);
    snapshot.phase_saved[idx] = phases.saved[idx];
  }

  keytrace_pending_snapshot_id = snapshot.snapshot_id;
  keytrace_snapshots[snapshot.snapshot_id] = snapshot;
}

void Internal::keytrace_decision (int lit) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  if (!lit || keytrace_pending_snapshot_id < 0)
    error ("keytrace decision without pending before-decision snapshot");
  if (keytrace_stack.empty () || keytrace_stack.back ().type != KEYTRACE_EVENT_DECISION)
    error ("keytrace decision assignment event missing");
  keytrace_pending_snapshot_id = -1;
}

void Internal::keytrace_backtrack (int from_level, int to_level, size_t before, size_t after, bool restart_trigger) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  (void) from_level;
  (void) before;
  (void) after;

  const bool save_trace_prefix =
      keytrace_trace_mode (this) && !restart_trigger && to_level == 0;
  if (save_trace_prefix) {
    for (const auto &event : keytrace_trace_prefix_before_final_unsat) {
      if (event.type != KEYTRACE_EVENT_DECISION || event.snapshot_id < 0)
        continue;
      const bool still_current = std::any_of (
          keytrace_stack.begin (), keytrace_stack.end (),
          [&event] (const KeyTraceEvent &current) {
            return current.snapshot_id == event.snapshot_id;
          });
      if (!still_current)
        keytrace_snapshots.erase (event.snapshot_id);
    }
    keytrace_trace_prefix_before_final_unsat = keytrace_stack;
    keytrace_trace_prefix_before_final_unsat_valid = true;
  }

  while (!keytrace_stack.empty () && keytrace_stack.back ().level > to_level) {
    const KeyTraceEvent event = keytrace_stack.back ();
    if (!save_trace_prefix && event.snapshot_id >= 0)
      keytrace_snapshots.erase (event.snapshot_id);
    keytrace_stack.pop_back ();
  }
  while (!keytrace_paper_stack.empty () &&
         keytrace_paper_stack.back ().level > to_level)
    keytrace_paper_stack.pop_back ();
  if (!restart_trigger) {
    KeyTracePaperEvent paper;
    paper.raw_index = ++keytrace_paper_raw_event_index;
    paper.type = 'B';
    paper.lit = 0;
    paper.level = to_level;
    keytrace_paper_stack.push_back (paper);
  }
  for (int idx = 1; idx < (int) keytrace_assignment_source_by_var.size (); idx++)
    if (!val (idx))
      keytrace_assignment_source_by_var[idx] = KEYTRACE_UNASSIGNED;
}

void Internal::keytrace_restart (int from_level, int to_level, size_t before, size_t after) {
  if (!keytrace_dump || !keytrace_search_started)
    return;
  (void) from_level;
  (void) to_level;
  (void) before;
  (void) after;
  // Paper-original KeyTrace uses restart-to-root semantics even when
  // state-aware KeyTrace preserves CaDiCaL's actual reuse-trail level.
  while (!keytrace_paper_stack.empty () &&
         keytrace_paper_stack.back ().level > 0)
    keytrace_paper_stack.pop_back ();
  keytrace_conflicts_at_last_restart = stats.conflicts;
  keytrace_decisions_at_last_restart = stats.decisions;
  keytrace_propagations_at_last_restart = stats.propagations.search;
}

static int sample_id_width (int64_t n) {
  int res = 6;
  int digits = 1;
  while (n >= 10)
    n /= 10, digits++;
  if (digits > res)
    res = digits;
  return res;
}

static std::string sample_file_name (const Internal *internal,
                                    int64_t id, int width) {
  std::ostringstream out;
  out << internal->keytrace_output_prefix << ".sample." << std::setfill ('0')
      << std::setw (width) << id << ".jsonl.gz";
  return out.str ();
}

static std::string sample_path (const Internal *internal, int64_t id,
                                int width) {
  return join_path (keytrace_dump_dir (internal),
                    sample_file_name (internal, id, width));
}

static void write_paper_keytrace (Internal *internal) {
  const std::string path = artifact_path (internal, "paper.jsonl");
  File *file = File::write (internal, path.c_str ());
  if (!file)
    internal->error ("can not write keytrace paper file '%s'", path.c_str ());
  put (file, "{\"type\":\"begin\",\"schema\":\"paper_original_keytrace_v1\"}");
  file->endl ();
  for (const auto &event : internal->keytrace_paper_stack) {
    put (file, "{\"type\":\"paper_keytrace_event\",\"raw_index\":");
    file->put ((int64_t) event.raw_index);
    put (file, ",\"event\":\"");
    if (event.type == 'B')
      put (file, "BT");
    else
      file->put (event.type);
    put (file, "\",\"lit\":");
    file->put ((int64_t) event.lit);
    put (file, ",\"level\":");
    file->put ((int64_t) event.level);
    put (file, "}");
    file->endl ();
  }
  put (file, "{\"type\":\"end\",\"events\":");
  file->put ((int64_t) internal->keytrace_paper_stack.size ());
  put (file, "}");
  file->endl ();
  delete file;
}

static void write_clause_literals (File *file,
                                  const std::vector<int> &lits) {
  for (size_t i = 0; i < lits.size (); i++) {
    if (i)
      file->put (' ');
    file->put ((int64_t) lits[i]);
  }
  if (!lits.empty ())
    file->put (' ');
  file->put ((int64_t) 0);
}

static void write_clause_stream (File *file,
                                 const std::vector<int> &lits) {
  write_clause_literals (file, lits);
  file->endl ();
}

static std::shared_ptr<const std::vector<std::vector<int>>>
replay_realtime_clause_deltas (Internal *internal, int64_t boundary) {
  std::map<int64_t, std::vector<int>> active;
  for (size_t i = 0; i < internal->keytrace_base_clauses.size (); i++)
    active[(int64_t) i + 1] = internal->keytrace_base_clauses[i];

  int64_t previous_event_id = 0;
  for (const auto &delta : internal->keytrace_clause_deltas) {
    if (delta.event_id <= previous_event_id)
      internal->error ("non-monotonic keytrace clause delta event id");
    previous_event_id = delta.event_id;
    if (delta.event_id > boundary)
      break;
    if (delta.type == 'a') {
      if (active.count (delta.artifact_clause_id))
        internal->error ("duplicate keytrace artifact clause id %" PRId64,
                         delta.artifact_clause_id);
      active[delta.artifact_clause_id] = delta.lits;
    } else if (delta.type == 'd') {
      const auto pos = active.find (delta.artifact_clause_id);
      if (pos == active.end ())
        internal->error ("delete of unknown keytrace artifact clause id %" PRId64,
                         delta.artifact_clause_id);
      active.erase (pos);
    } else if (delta.type == 'r') {
      const auto old_pos = active.find (delta.old_artifact_clause_id);
      if (old_pos == active.end ())
        internal->error ("replace of unknown keytrace artifact clause id %" PRId64,
                         delta.old_artifact_clause_id);
      active.erase (old_pos);
      if (active.count (delta.artifact_clause_id))
        internal->error ("duplicate keytrace artifact clause id %" PRId64,
                         delta.artifact_clause_id);
      active[delta.artifact_clause_id] = delta.lits;
    } else
      internal->error ("invalid keytrace clause delta type '%c'", delta.type);
  }

  std::vector<std::vector<int>> cnf;
  for (const auto &entry : active)
    cnf.push_back (entry.second);
  std::sort (cnf.begin (), cnf.end ());
  return std::make_shared<const std::vector<std::vector<int>>> (cnf);
}

static std::string keytrace_clause_to_string (const std::vector<int> &clause) {
  std::ostringstream out;
  for (size_t i = 0; i < clause.size (); i++) {
    if (i)
      out << ' ';
    out << clause[i];
  }
  return out.str ();
}

static void keytrace_first_cnf_difference (
    const std::vector<std::vector<int>> &direct,
    const std::vector<std::vector<int>> &replayed,
    std::vector<int> &direct_only,
    std::vector<int> &replay_only) {
  size_t i = 0, j = 0;
  while (i < direct.size () || j < replayed.size ()) {
    if (i == direct.size ()) {
      replay_only = replayed[j];
      return;
    }
    if (j == replayed.size ()) {
      direct_only = direct[i];
      return;
    }
    if (direct[i] == replayed[j])
      i++, j++;
    else {
      direct_only = direct[i];
      replay_only = replayed[j];
      return;
    }
  }
}

struct MaterializedKeyTraceSample {
  KeyTraceEvent event;
  KeyTraceSnapshot snapshot;
  std::shared_ptr<const std::vector<std::vector<int>>> cnf;
};

static std::vector<MaterializedKeyTraceSample> materialize_surviving_samples (
    Internal *internal, const std::vector<KeyTraceEvent> &surviving) {
  std::vector<MaterializedKeyTraceSample> samples;
  for (const auto &event : surviving) {
    const auto pos = internal->keytrace_snapshots.find (event.snapshot_id);
    if (pos == internal->keytrace_snapshots.end ())
      internal->error ("surviving keytrace decision lacks snapshot");

    MaterializedKeyTraceSample sample;
    sample.event = event;
    sample.snapshot = pos->second;
    if (internal->keytrace_cnf_mode == "sample-first") {
      if (!sample.snapshot.cnf)
        internal->error ("sample-first keytrace snapshot lacks direct CNF");
      sample.cnf = sample.snapshot.cnf;
    } else if (internal->keytrace_cnf_mode == "clause-delta") {
      sample.cnf = replay_realtime_clause_deltas (
          internal, sample.snapshot.artifact_replay_event_id);
    } else if (internal->keytrace_cnf_mode == "sample-delta-reply") {
      if (!sample.snapshot.cnf)
        internal->error ("sample-delta-reply snapshot lacks direct CNF");
      sample.cnf = sample.snapshot.cnf;
      const auto replayed = replay_realtime_clause_deltas (
          internal, sample.snapshot.artifact_replay_event_id);
      if (*sample.cnf != *replayed) {
        std::vector<int> direct_only, replay_only;
        keytrace_first_cnf_difference (*sample.cnf, *replayed, direct_only,
                                       replay_only);
        internal->error (
            "sample-delta-reply direct snapshot and replay CNF mismatch at snapshot %" PRId64
            " boundary %" PRId64 " direct_clauses %" PRId64 " replay_clauses %" PRId64
            " direct_only [%s] replay_only [%s]",
            sample.snapshot.snapshot_id,
            sample.snapshot.artifact_replay_event_id,
            (int64_t) sample.cnf->size (), (int64_t) replayed->size (),
            keytrace_clause_to_string (direct_only).c_str (),
            keytrace_clause_to_string (replay_only).c_str ());
      }
    } else
      internal->error ("invalid keytrace CNF mode '%s'",
                       internal->keytrace_cnf_mode.c_str ());
    samples.push_back (sample);
  }
  return samples;
}

static void write_realtime_clause_deltas (Internal *internal, File *delta) {
  for (const auto &event : internal->keytrace_clause_deltas) {
    delta->put (event.type);
    delta->put (' ');
    delta->put (event.event_id);
    delta->put (' ');
    delta->put (event.artifact_clause_id);
    delta->put (' ');
    delta->put (event.old_artifact_clause_id);
    delta->put (' ');
    write_clause_stream (delta, event.lits);
  }
}

static void write_artifacts (Internal *internal,
                             const std::vector<MaterializedKeyTraceSample> &samples,
                             int width, int res) {
  if (!keytrace_clause_delta_mode (internal))
    return;

  const std::string base_path = artifact_path (internal, "search_base.cnf.gz");
  File *base = File::write (internal, base_path.c_str ());
  if (!base)
    internal->error ("can not write keytrace base CNF file '%s'",
                     base_path.c_str ());
  for (const auto &clause : internal->keytrace_base_clauses)
    write_clause_stream (base, clause);
  delete base;

  const std::string delta_path = artifact_path (internal, "clauses.delta.gz");
  File *delta = File::write (internal, delta_path.c_str ());
  if (!delta)
    internal->error ("can not write keytrace clause delta file '%s'",
                     delta_path.c_str ());
  write_realtime_clause_deltas (internal, delta);
  delete delta;

  const std::string index_path = artifact_path (internal, "keytrace.jsonl");
  File *index = File::write (internal, index_path.c_str ());
  if (!index)
    internal->error ("can not write keytrace artifact index '%s'",
                     index_path.c_str ());
  put (index, "{\"type\":\"begin\",\"schema\":\"keytrace_artifacts_v1_6\"}");
  index->endl ();
  put (index, "{\"type\":\"keytrace_summary\",\"result\":");
  index->put ((int64_t) res);
  put (index, ",\"run_result\":\"");
  put (index, keytrace_run_result (res));
  put (index, "\",\"profile_mode\":\"");
  put (index, internal->keytrace_profile_mode);
  put (index, "\"");
  put (index, ",\"samples\":");
  index->put ((int64_t) samples.size ());
  put (index, ",\"sample_top_k\":");
  index->put ((int64_t) internal->keytrace_sample_top_k);
  put (index, ",\"cnf_mode\":\"");
  put (index, internal->keytrace_cnf_mode);
  put (index, "\"");
  put (index, ",\"search_base_cnf\":\"");
  put (index, prefixed_keytrace_file_name (internal, "search_base.cnf.gz"));
  put (index, "\"");
  put (index, ",\"clauses_delta\":\"");
  put (index, prefixed_keytrace_file_name (internal, "clauses.delta.gz"));
  put (index, "\"");
  put (index, ",\"paper\":\"");
  put (index, prefixed_keytrace_file_name (internal, "paper.jsonl"));
  put (index, "\"");
  put (index, ",\"learnts_enabled\":");
  index->put (internal->keytrace_dump_learnts ? 1 : 0);
  put (index, ",\"base_delta_enabled\":1");
  put (index, ",\"inprocessing_mode\":\"");
  put (index, keytrace_inprocessing_mode (internal));
  put (index, "\"");
  put (index, ",\"clause_delta_events\":");
  index->put ((int64_t) internal->keytrace_clause_deltas.size ());
  put (index, "}");
  index->endl ();
  for (size_t i = 0; i < samples.size (); i++) {
    const auto &sample = samples[i];
    put (index, "{\"type\":\"sample_index\",\"sample_id\":");
    index->put ((int64_t) i);
    put (index, ",\"snapshot_id\":");
    index->put (sample.snapshot.snapshot_id);
    put (index, ",\"decision_event_id\":");
    index->put (sample.event.event_id);
    put (index, ",\"keytrace_pos\":");
    index->put ((int64_t) i);
    put (index, ",\"label_lit\":");
    index->put ((int64_t) sample.event.lit);
    put (index, ",\"artifact_replay_event_id\":");
    index->put (sample.snapshot.artifact_replay_event_id);
    put (index, ",\"assignment_source\":\"");
    put (index, source_name (sample.event.assignment_source));
    put (index, "\",\"sample_file\":\"../");
    put (index, sample_file_name (internal, i, width));
    put (index, "\"}");
    index->endl ();
  }
  put (index, "{\"type\":\"end\"}");
  index->endl ();
  delete index;

  if (internal->keytrace_dump_debug) {
    int64_t add_count = 0, delete_count = 0, replace_count = 0;
    for (const auto &event : internal->keytrace_clause_deltas) {
      if (event.type == 'a') add_count++;
      else if (event.type == 'd') delete_count++;
      else if (event.type == 'r') replace_count++;
    }
    const std::string debug_path = artifact_path (internal, "debug.jsonl");
    File *debug = File::write (internal, debug_path.c_str ());
    if (!debug)
      internal->error ("can not write keytrace debug file '%s'",
                       debug_path.c_str ());
    put (debug, "{\"type\":\"debug_summary\",\"events\":");
    debug->put (internal->keytrace_event_id);
    put (debug, ",\"snapshots\":");
    debug->put (internal->keytrace_snapshot_id);
    put (debug, ",\"cnf_mode\":\"");
    put (debug, internal->keytrace_cnf_mode);
    put (debug, "\",\"clause_delta_events\":");
    debug->put ((int64_t) internal->keytrace_clause_deltas.size ());
    put (debug, ",\"add_count\":");
    debug->put (add_count);
    put (debug, ",\"delete_count\":");
    debug->put (delete_count);
    put (debug, ",\"replace_count\":");
    debug->put (replace_count);
    put (debug, "}");
    debug->endl ();
    delete debug;
  }
}

static void write_sample (Internal *internal, const std::string &path, int64_t sample_id, int64_t keytrace_pos, const KeyTraceEvent &event, const KeyTraceSnapshot &snapshot, const std::vector<std::vector<int>> &sample_cnf, const char *run_result) {
  if (strcmp (run_result, "SAT") && strcmp (run_result, "UNSAT"))
    internal->error ("invalid keytrace sample run result '%s'", run_result);
  if (event.type != KEYTRACE_EVENT_DECISION ||
      event.assignment_source != KEYTRACE_DECISION)
    internal->error ("keytrace sample label is not an ordinary decision");
  if (event.snapshot_id != snapshot.snapshot_id)
    internal->error ("keytrace decision snapshot mismatch");
  const int label_lit = event.lit;
  const int label_idx = abs (label_lit);
  if (!label_lit || label_idx > internal->max_var)
    internal->error ("invalid keytrace label literal %d", label_lit);
  if (label_idx < (int) snapshot.assignment_value.size () && snapshot.assignment_value[label_idx])
    internal->error ("keytrace label literal %d already assigned in snapshot", label_lit);
  if (snapshot.assignment_value.empty () && internal->max_var > 0)
    internal->error ("keytrace snapshot state payload missing for sample");

  File *file = File::write (internal, path.c_str ());
  if (!file)
    internal->error ("can not write keytrace sample file '%s'", path.c_str ());
  put (file, "{\"type\":\"sample_meta\",\"schema\":\"keytrace_state_aware_v1_6\",\"sample_id\":");
  file->put (sample_id);
  put (file, ",\"snapshot_id\":");
  file->put (snapshot.snapshot_id);
  put (file, ",\"decision_event_id\":");
  file->put (event.event_id);
  put (file, ",\"keytrace_pos\":");
  file->put (keytrace_pos);
  put (file, ",\"num_vars\":");
  file->put ((int64_t) internal->max_var);
  put (file, ",\"num_clauses\":");
  file->put ((int64_t) sample_cnf.size ());
  put (file, ",\"literal_space\":\"internal\",\"cnf_mode\":\"");
  put (file, internal->keytrace_cnf_mode);
  put (file, "\",\"profile_mode\":\"");
  put (file, internal->keytrace_profile_mode);
  put (file, "\",\"run_result\":\"");
  put (file, run_result);
  put (file, "\"}");
  file->endl ();

  put (file, "{\"type\":\"cnf\",\"clauses\":[");
  for (size_t i = 0; i < sample_cnf.size (); i++) {
    if (i)
      file->put (',');
    put_int_array (file, sample_cnf[i], 0);
  }
  put (file, "]}");
  file->endl ();

  put (file, "{\"type\":\"solver_state\",\"decision_level\":");
  file->put ((int64_t) snapshot.decision_level);
  put (file, ",\"trail_size\":");
  file->put ((int64_t) snapshot.trail_size);
  put (file, ",\"num_assigned\":");
  file->put ((int64_t) snapshot.num_assigned);
  put (file, ",\"conflicts\":");
  file->put (snapshot.conflicts);
  put (file, ",\"decisions\":");
  file->put (snapshot.decisions);
  put (file, ",\"propagations\":");
  file->put (snapshot.propagations);
  put (file, ",\"restarts\":");
  file->put (snapshot.restarts);
  put (file, ",\"conflicts_since_restart\":");
  file->put (snapshot.conflicts_since_restart);
  put (file, ",\"decisions_since_restart\":");
  file->put (snapshot.decisions_since_restart);
  put (file, ",\"propagations_since_restart\":");
  file->put (snapshot.propagations_since_restart);
  put (file, ",\"last_clause_event_id\":");
  file->put (snapshot.last_clause_event_id);
  put (file, ",\"artifact_replay_event_id\":");
  file->put (snapshot.artifact_replay_event_id);
  put (file, ",\"assignment_value\":");
  put_char_array (file, snapshot.assignment_value, 1);
  put (file, ",\"assignment_level\":");
  put_int_array (file, snapshot.assignment_level, 1);
  put (file, ",\"assignment_source\":");
  put_char_array (file, snapshot.assignment_source, 1);
  put (file, ",\"trail_position\":");
  put_int_array (file, snapshot.trail_position, 1);
  put (file, ",\"activity\":");
  put_float_array (file, snapshot.activity, 1);
  put (file, ",\"phase_saved\":");
  put_char_array (file, snapshot.phase_saved, 1);
  put (file, "}");
  file->endl ();

  put (file, "{\"type\":\"label\",\"label_lit\":");
  file->put ((int64_t) label_lit);
  put (file, ",\"assignment_source\":\"DECISION\"}");
  file->endl ();
  delete file;
}

void Internal::keytrace_finish (int res) {
  if (!keytrace_dump)
    return;

  const char *run_result = keytrace_run_result (res);
  std::vector<KeyTraceEvent> surviving;
  if (res == 10 || res == 20) {
    const std::vector<KeyTraceEvent> *source = &keytrace_stack;
    if (keytrace_trace_mode (this) && res == 20 &&
        keytrace_trace_prefix_before_final_unsat_valid)
      source = &keytrace_trace_prefix_before_final_unsat;
    for (const auto &event : *source)
      if (event.type == KEYTRACE_EVENT_DECISION &&
          event.assignment_source == KEYTRACE_DECISION)
        surviving.push_back (event);
  }
  if (keytrace_sample_top_k > 0 &&
      (int) surviving.size () > keytrace_sample_top_k)
    surviving.resize (keytrace_sample_top_k);

  const std::string out_dir = keytrace_dump_dir (this);
  if (!mkdir_p (out_dir))
    error ("can not create keytrace dump directory '%s'", out_dir.c_str ());
  if (!mkdir_p (artifacts_dir (this)))
    error ("can not create keytrace artifacts directory '%s'",
           artifacts_dir (this).c_str ());

  const int width = sample_id_width (surviving.empty () ? 0 :
                                     surviving.size () - 1);
  const std::vector<MaterializedKeyTraceSample> samples =
      materialize_surviving_samples (this, surviving);
  for (size_t i = 0; i < samples.size (); i++) {
    const std::string path = sample_path (this, i, width);
    write_sample (this, path, i, i, samples[i].event, samples[i].snapshot,
                  *samples[i].cnf, run_result);
  }
  write_paper_keytrace (this);
  write_artifacts (this, samples, width, res);
  keytrace_search_started = false;
}

} // namespace CaDiCaL
