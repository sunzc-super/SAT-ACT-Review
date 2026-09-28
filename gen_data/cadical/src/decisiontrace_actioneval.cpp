#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace CaDiCaL {

namespace {

enum ProposalSource {
  PROPOSAL_NATIVE = 1,
  PROPOSAL_NATIVE_OPPOSITE = 2,
  PROPOSAL_ACTIVITY_PHASE = 4,
  PROPOSAL_ACTIVITY_OPPOSITE = 8,
  PROPOSAL_JW = 16,
  PROPOSAL_RANDOM = 32
};

struct Proposal {
  int external_literal;
  unsigned sources;
  double activity;
  double jw;
  uint64_t random_key;
};

static void put_json_string (FILE *file, const std::string &value) {
  fputc ('"', file);
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"': fputs ("\\\"", file); break;
    case '\\': fputs ("\\\\", file); break;
    case '\b': fputs ("\\b", file); break;
    case '\f': fputs ("\\f", file); break;
    case '\n': fputs ("\\n", file); break;
    case '\r': fputs ("\\r", file); break;
    case '\t': fputs ("\\t", file); break;
    default:
      if (ch < 0x20)
        fprintf (file, "\\u%04x", (unsigned) ch);
      else
        fputc (ch, file);
    }
  }
  fputc ('"', file);
}

static void put_hash (FILE *file, uint64_t hash) {
  fprintf (file, "\"%016llx\"", (unsigned long long) hash);
}

static uint64_t mix64 (uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

static void hash_word (uint64_t &hash, uint64_t word) {
  for (unsigned i = 0; i != 8; ++i) {
    hash ^= (unsigned char) (word & 0xffu);
    hash *= 1099511628211ULL;
    word >>= 8;
  }
}

static uint64_t hash_text (const std::string &text) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char ch : text) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  return hash;
}

static bool within_limit (int64_t value, int64_t limit) {
  return !limit || value <= limit;
}

static int external_literal (Internal *internal, int ilit) {
  return ilit ? internal->externalize (ilit) : 0;
}

static int existing_internal_literal (Internal *internal, int elit) {
  if (!elit || elit == INT_MIN || !internal->external)
    return 0;
  const int eidx = abs (elit);
  External *external = internal->external;
  if (eidx > external->max_var || (size_t) eidx >= external->e2i.size ())
    return 0;
  int ilit = external->e2i[eidx];
  if (elit < 0)
    ilit = -ilit;
  return ilit;
}

static bool candidate_internal_variable (Internal *internal, int idx) {
  if (idx <= 0 || idx > internal->max_var || internal->val (idx) ||
      !internal->flags (idx).active ())
    return false;
  const int elit = internal->externalize (idx);
  if (!elit || abs (elit) > internal->external->max_var ||
      !internal->external->observed (abs (elit)))
    return false;
  return abs (existing_internal_literal (internal, elit)) == idx;
}

struct StateFingerprint {
  uint64_t assignment;
  uint64_t trail;
  uint64_t clause_canonical;
  uint64_t clause_order;
  uint64_t eligibility;
};

static uint64_t eligibility_hash (Internal *internal) {
  uint64_t hash = 1469598103934665603ULL;
  hash_word (hash, (uint64_t) internal->external->max_var);
  for (int eidx = 1; eidx <= internal->external->max_var; ++eidx) {
    const int ilit = existing_internal_literal (internal, eidx);
    hash_word (hash, ilit && candidate_internal_variable (internal, abs (ilit))
                         ? 1u
                         : 0u);
  }
  return hash;
}

static StateFingerprint state_fingerprint (Internal *internal) {
  StateFingerprint result;
  result.assignment = 1469598103934665603ULL;
  result.trail = 1469598103934665603ULL;
  result.clause_order = 1469598103934665603ULL;
  result.eligibility = eligibility_hash (internal);

  hash_word (result.assignment, (uint64_t) internal->max_var);
  for (int idx = 1; idx <= internal->max_var; ++idx) {
    hash_word (result.assignment,
               (uint64_t) (int64_t) internal->externalize (idx));
    hash_word (result.assignment,
               (uint64_t) (int64_t) internal->val (idx));
    hash_word (result.assignment,
               (uint64_t) (int64_t) (internal->val (idx)
                                          ? internal->var (idx).level
                                          : -1));
  }

  hash_word (result.trail, (uint64_t) internal->trail.size ());
  for (const int ilit : internal->trail)
    hash_word (result.trail,
               (uint64_t) (int64_t) external_literal (internal, ilit));

  uint64_t canonical_xor = 0;
  uint64_t canonical_sum = 0;
  uint64_t clause_count = 0;
  for (Clause *clause : internal->clauses) {
    if (!clause || clause->garbage)
      continue;
    ++clause_count;
    hash_word (result.clause_order, (uint64_t) clause->redundant);
    hash_word (result.clause_order, (uint64_t) clause->size);
    std::vector<int> sorted;
    sorted.reserve (clause->size);
    for (const int ilit : *clause) {
      hash_word (result.clause_order, (uint64_t) (int64_t) ilit);
      sorted.push_back (ilit);
    }
    std::sort (sorted.begin (), sorted.end ());
    uint64_t clause_hash = 1469598103934665603ULL;
    hash_word (clause_hash, (uint64_t) clause->redundant);
    hash_word (clause_hash, (uint64_t) sorted.size ());
    for (const int ilit : sorted)
      hash_word (clause_hash, (uint64_t) (int64_t) ilit);
    const uint64_t mixed = mix64 (clause_hash);
    canonical_xor ^= mixed;
    canonical_sum += mixed;
  }
  hash_word (result.clause_order, clause_count);
  result.clause_canonical = 1469598103934665603ULL;
  hash_word (result.clause_canonical, clause_count);
  hash_word (result.clause_canonical, canonical_xor);
  hash_word (result.clause_canonical, canonical_sum);
  return result;
}

static int count_candidates (Internal *internal) {
  int result = 0;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    result += candidate_internal_variable (internal, idx);
  return result;
}

static int64_t cheap_state_clause_count (Internal *internal) {
  return internal->stats.current.irredundant +
         internal->stats.current.redundant;
}

static bool prefix_budget_ok (Internal *internal) {
  const DecisionTraceOptions &o = internal->decisiontrace_options;
  const DecisionTraceRuntime &r = internal->decisiontrace_runtime;
  const int64_t conflicts =
      r.callback_pending ? r.pending_conflicts : internal->stats.conflicts;
  const int64_t decisions =
      r.callback_pending ? r.pending_decisions : internal->stats.decisions;
  const int64_t propagations = r.callback_pending
                                   ? r.pending_propagations
                                   : internal->stats.propagations.search;
  return within_limit (conflicts, o.max_prefix_conflicts) &&
         within_limit (decisions, o.max_prefix_decisions) &&
         within_limit (propagations, o.max_prefix_propagations) &&
         within_limit (r.callback_attempt_index, o.max_prefix_callbacks);
}

static bool prefix_budget_has_room (Internal *internal) {
  const DecisionTraceOptions &o = internal->decisiontrace_options;
  const DecisionTraceRuntime &r = internal->decisiontrace_runtime;
  return (!o.max_prefix_conflicts ||
          internal->stats.conflicts < o.max_prefix_conflicts) &&
         (!o.max_prefix_decisions ||
          internal->stats.decisions < o.max_prefix_decisions) &&
         (!o.max_prefix_propagations ||
          internal->stats.propagations.search < o.max_prefix_propagations) &&
         (!o.max_prefix_callbacks ||
          r.callback_attempt_index < o.max_prefix_callbacks);
}

static bool cheap_eligible (Internal *internal, int candidates,
                            int64_t &state_nodes) {
  const DecisionTraceOptions &o = internal->decisiontrace_options;
  state_nodes = 2 * (int64_t) internal->max_var +
                cheap_state_clause_count (internal);
  return candidates > 0 && within_limit (state_nodes, o.max_state_nodes);
}

static void write_schema (Internal *internal) {
  FILE *f = internal->decisiontrace_runtime.output;
  fputs ("{\"type\":\"schema\",\"name\":\"DecisionTrace-ActionEval-raw\","
         "\"version\":1,\"records\":{", f);
  fputs ("\"tr\":[\"instance\",\"callback\",\"raw_decision\","
         "\"eligible_state\",\"level\",\"trail\",\"conflicts\","
         "\"decisions\",\"propagations\",\"restarts\",\"candidates\","
         "\"state_nodes_preflight\",\"prefix_hash\",\"eligibility_hash\","
         "\"native_external\","
         "\"native_internal\"],", f);
  fputs ("\"st\":[\"instance\",\"eligible_state\",\"callback\","
         "\"raw_decision\","
         "\"level\",\"trail\",\"conflicts\",\"decisions\","
         "\"propagations\",\"restarts\",\"prefix_hash\",\"native_external\","
         "\"internal_variables\",\"external_variables\",\"n_clauses\","
         "\"n_occurrences\",\"assignment_hash\",\"trail_hash\","
         "\"clause_canonical_hash\",\"clause_order_hash\","
         "\"eligibility_hash\",\"i2e\","
         "\"eligibility_bits_hex\",\"assignment_value\","
         "\"assignment_level\",\"assignment_source\",\"trail_position\","
         "\"activity\",\"saved_phase\",\"clause_offsets\","
         "\"clause_literals_internal\"],", f);
  fputs ("\"sl\":[\"instance\",\"eligible_state\",\"callback\","
         "\"raw_decision\",\"level\",\"trail\",\"conflicts\","
         "\"decisions\",\"propagations\",\"restarts\",\"prefix_hash\","
         "\"native_external\",\"assignment_hash\",\"trail_hash\","
         "\"clause_canonical_hash\",\"clause_order_hash\","
         "\"eligibility_hash\",\"state_nodes\",\"literal_occurrences\","
         "\"max_clause_length\",\"estimated_payload_bytes\"],", f);
  fputs ("\"pr\":[\"instance\",\"eligible_state\",\"actions\"],"
         "\"ev\":[\"instance\",\"eligible_state\",\"callback\","
         "\"raw_decision\",\"requested_external\","
         "\"target_reached\",\"locator_match\",\"action_eligible\",\"action_applied\","
         "\"prefix_hash\",\"native_external\",\"applied_external\","
         "\"status\",\"censored\",\"horizon_hit\",\"delta_conflicts\","
         "\"delta_decisions\",\"delta_propagations\",\"delta_restarts\"],"
         "\"out\":[\"instance\",\"mode\",\"status\",\"conflicts\","
         "\"decisions\",\"propagations\",\"restarts\",\"callbacks\","
         "\"eligible_states\",\"materialized\",\"materialization_failures\"],"
         "\"sf\":[\"instance\",\"eligible_state\",\"reason\","
         "\"state_nodes\",\"literal_occurrences\",\"max_clause_length\","
         "\"estimated_payload_bytes\"]", f);
  fputs ("},\"proposal_source_bits\":{\"native\":1,"
         "\"native_opposite\":2,\"activity_phase\":4,"
         "\"activity_opposite\":8,\"jw\":16,\"random\":32}}\n", f);
}

static void emit_trace_record (Internal *internal, int candidates,
                               int64_t state_nodes, int native_internal) {
  DecisionTraceRuntime &r = internal->decisiontrace_runtime;
  FILE *f = r.output;
  fputs ("[\"tr\",", f);
  put_json_string (f, internal->decisiontrace_options.instance_id);
  fprintf (f, ",%lld,%lld,%lld,%d,%zu,%lld,%lld,%lld,%lld,%d,%lld,",
           (long long) r.callback_attempt_index,
           (long long) r.raw_decision_index,
           (long long) r.eligible_state_index, r.pending_level,
           (size_t) r.pending_trail_size, (long long) r.pending_conflicts,
           (long long) r.pending_decisions,
           (long long) r.pending_propagations,
           (long long) r.pending_restarts, candidates,
           (long long) state_nodes);
  put_hash (f, r.pending_prefix_hash);
  fputc (',', f);
  put_hash (f, eligibility_hash (internal));
  fprintf (f, ",%d,%d]\n", external_literal (internal, native_internal),
           native_internal);
}

static void add_proposal (std::vector<Proposal> &actions,
                          std::unordered_map<int, size_t> &by_literal,
                          int elit, unsigned source, double activity,
                          double jw, uint64_t random_key) {
  if (!elit)
    return;
  const auto found = by_literal.find (elit);
  if (found != by_literal.end ()) {
    Proposal &p = actions[found->second];
    p.sources |= source;
    if (std::isfinite (activity))
      p.activity = activity;
    if (std::isfinite (jw))
      p.jw = jw;
    if (random_key)
      p.random_key = random_key;
    return;
  }
  Proposal p;
  p.external_literal = elit;
  p.sources = source;
  p.activity = std::isfinite (activity) ? activity : 0.0;
  p.jw = std::isfinite (jw) ? jw : 0.0;
  p.random_key = random_key;
  by_literal[elit] = actions.size ();
  actions.push_back (p);
}

static std::vector<Proposal> collect_proposals (Internal *internal,
                                                int native_internal) {
  const DecisionTraceOptions &o = internal->decisiontrace_options;
  const bool variable_mode = o.max_actions_mode == "variable";
  const int n = internal->max_var;
  const double absent_score = std::numeric_limits<double>::quiet_NaN ();
  std::vector<Proposal> result;
  std::unordered_map<int, size_t> by_literal;
  std::vector<int> variable_sources[3];

  const int native_external = external_literal (internal, native_internal);
  if (native_internal &&
      candidate_internal_variable (internal, abs (native_internal))) {
    add_proposal (result, by_literal, native_external, PROPOSAL_NATIVE,
                  internal->score (abs (native_internal)), absent_score, 0);
    add_proposal (result, by_literal, -native_external,
                  PROPOSAL_NATIVE_OPPOSITE,
                  internal->score (abs (native_internal)), absent_score, 0);
  }

  std::vector<std::pair<double, int>> activity;
  activity.reserve (n);
  for (int idx = 1; idx <= n; ++idx)
    if (candidate_internal_variable (internal, idx))
      activity.push_back (std::make_pair (internal->score (idx), idx));
  std::sort (activity.begin (), activity.end (),
             [] (const std::pair<double, int> &a,
                 const std::pair<double, int> &b) {
               if (a.first != b.first)
                 return a.first > b.first;
               return a.second < b.second;
             });
  const int activity_count =
      std::min<int> (std::max (0, o.activity_top_k), activity.size ());
  for (int i = 0; i < activity_count; ++i) {
    const int idx = activity[i].second;
    int phase = internal->phases.saved[idx];
    if (!phase)
      phase = internal->opts.phase ? 1 : -1;
    const int preferred = phase * idx;
    if (variable_mode)
      variable_sources[0].push_back (
          external_literal (internal, preferred));
    add_proposal (result, by_literal, external_literal (internal, preferred),
                  PROPOSAL_ACTIVITY_PHASE, activity[i].first, absent_score, 0);
    add_proposal (result, by_literal, external_literal (internal, -preferred),
                  PROPOSAL_ACTIVITY_OPPOSITE, activity[i].first, absent_score,
                  0);
  }

  std::vector<double> jw (2u * (n + 1), 0.0);
  for (Clause *clause : internal->clauses) {
    if (!clause || clause->garbage)
      continue;
    bool satisfied = false;
    int residual_size = 0;
    for (const int lit : *clause) {
      const signed char value = internal->val (lit);
      if (value > 0) {
        satisfied = true;
        break;
      }
      if (!value && internal->flags (lit).active ())
        ++residual_size;
    }
    if (satisfied || !residual_size)
      continue;
    const double weight = std::ldexp (1.0, -std::min (residual_size, 1022));
    for (const int lit : *clause)
      if (!internal->val (lit) && internal->flags (lit).active ()) {
        const size_t pos = 2u * (size_t) abs (lit) + (lit < 0);
        jw[pos] += weight;
      }
  }
  if (variable_mode) {
    std::vector<std::pair<double, int>> jw_variables;
    jw_variables.reserve (n);
    for (int idx = 1; idx <= n; ++idx)
      if (candidate_internal_variable (internal, idx))
        jw_variables.push_back (
            std::make_pair (std::max (jw[2u * idx], jw[2u * idx + 1u]), idx));
    std::sort (jw_variables.begin (), jw_variables.end (),
               [] (const std::pair<double, int> &a,
                   const std::pair<double, int> &b) {
                 if (a.first != b.first)
                   return a.first > b.first;
                 return a.second < b.second;
               });
    const int jw_count =
        std::min<int> (std::max (0, o.jw_top_k), jw_variables.size ());
    for (int i = 0; i < jw_count; ++i) {
      const int idx = jw_variables[i].second;
      const int preferred = jw[2u * idx] >= jw[2u * idx + 1u] ? idx : -idx;
      variable_sources[1].push_back (
          external_literal (internal, preferred));
      add_proposal (result, by_literal,
                    external_literal (internal, preferred), PROPOSAL_JW,
                    internal->score (idx),
                    jw[2u * idx + (preferred < 0)], 0);
      add_proposal (result, by_literal,
                    external_literal (internal, -preferred), PROPOSAL_JW,
                    internal->score (idx),
                    jw[2u * idx + (preferred > 0)], 0);
    }
  } else {
    std::vector<std::pair<double, int>> jw_ranked;
    jw_ranked.reserve (2u * n);
    for (int idx = 1; idx <= n; ++idx)
      if (candidate_internal_variable (internal, idx)) {
        jw_ranked.push_back (std::make_pair (jw[2u * idx], idx));
        jw_ranked.push_back (std::make_pair (jw[2u * idx + 1u], -idx));
      }
    std::sort (jw_ranked.begin (), jw_ranked.end (),
               [] (const std::pair<double, int> &a,
                   const std::pair<double, int> &b) {
                 if (a.first != b.first)
                   return a.first > b.first;
                 if (abs (a.second) != abs (b.second))
                   return abs (a.second) < abs (b.second);
                 return a.second > b.second;
               });
    const int jw_count =
        std::min<int> (std::max (0, o.jw_top_k), jw_ranked.size ());
    for (int i = 0; i < jw_count; ++i)
      add_proposal (result, by_literal,
                    external_literal (internal, jw_ranked[i].second),
                    PROPOSAL_JW, internal->score (abs (jw_ranked[i].second)),
                    jw_ranked[i].first, 0);
  }

  const uint64_t salt = mix64 (
      o.seed ^ hash_text (o.instance_id) ^
      (uint64_t) internal->decisiontrace_runtime.eligible_state_index);
  if (variable_mode) {
    std::vector<std::pair<uint64_t, int>> random_variables;
    random_variables.reserve (n);
    for (int idx = 1; idx <= n; ++idx)
      if (candidate_internal_variable (internal, idx)) {
        const int base_external = external_literal (internal, idx);
        const uint64_t base = (uint64_t) abs (base_external);
        random_variables.push_back (
            std::make_pair (mix64 (salt ^ base), base_external));
      }
    std::sort (random_variables.begin (), random_variables.end (),
               [] (const std::pair<uint64_t, int> &a,
                   const std::pair<uint64_t, int> &b) {
                 if (a.first != b.first)
                   return a.first < b.first;
                 return abs (a.second) < abs (b.second);
               });
    const int random_count =
        std::min<int> (std::max (0, o.random_top_k),
                       random_variables.size ());
    for (int i = 0; i < random_count; ++i) {
      const int elit = abs (random_variables[i].second);
      const int ilit = existing_internal_literal (internal, elit);
      variable_sources[2].push_back (elit);
      add_proposal (result, by_literal, elit, PROPOSAL_RANDOM,
                    ilit ? internal->score (abs (ilit)) : 0, absent_score,
                    random_variables[i].first);
      add_proposal (result, by_literal, -elit, PROPOSAL_RANDOM,
                    ilit ? internal->score (abs (ilit)) : 0, absent_score,
                    random_variables[i].first);
    }
  } else {
    std::vector<std::pair<uint64_t, int>> random_ranked;
    random_ranked.reserve (2u * n);
    for (int idx = 1; idx <= n; ++idx)
      if (candidate_internal_variable (internal, idx)) {
        const int base_external = external_literal (internal, idx);
        const uint64_t base = (uint64_t) abs (base_external);
        random_ranked.push_back (
            std::make_pair (mix64 (salt ^ (2u * base)), base_external));
        random_ranked.push_back (
            std::make_pair (mix64 (salt ^ (2u * base + 1u)), -base_external));
      }
    std::sort (random_ranked.begin (), random_ranked.end (),
               [] (const std::pair<uint64_t, int> &a,
                   const std::pair<uint64_t, int> &b) {
                 if (a.first != b.first)
                   return a.first < b.first;
                 return a.second < b.second;
               });
    const int random_count =
        std::min<int> (std::max (0, o.random_top_k), random_ranked.size ());
    for (int i = 0; i < random_count; ++i) {
      const int ilit =
          existing_internal_literal (internal, random_ranked[i].second);
      add_proposal (result, by_literal, random_ranked[i].second,
                    PROPOSAL_RANDOM,
                    ilit ? internal->score (abs (ilit)) : 0, absent_score,
                    random_ranked[i].first);
    }
  }

  if (variable_mode) {
    std::unordered_set<int> variables;
    for (const Proposal &proposal : result)
      variables.insert (abs (proposal.external_literal));
    if (o.max_actions <= 0 || (int) variables.size () <= o.max_actions)
      return result;

    std::vector<Proposal> ordered;
    ordered.reserve (2u * o.max_actions);
    std::unordered_set<int> selected_variables;
    const auto select_variable = [&] (const Proposal &proposal) {
      const int variable = abs (proposal.external_literal);
      if ((int) selected_variables.size () >= o.max_actions ||
          !selected_variables.insert (variable).second)
        return false;
      ordered.push_back (proposal);
      const auto opposite = by_literal.find (-proposal.external_literal);
      if (opposite != by_literal.end ())
        ordered.push_back (result[opposite->second]);
      return true;
    };
    for (const Proposal &proposal : result)
      if (proposal.sources & (PROPOSAL_NATIVE | PROPOSAL_NATIVE_OPPOSITE))
        select_variable (proposal);

    size_t positions[] = {0, 0, 0};
    while ((int) selected_variables.size () < o.max_actions) {
      bool progress = false;
      for (size_t source = 0; source != 3; ++source) {
        while (positions[source] < variable_sources[source].size ()) {
          const int literal = variable_sources[source][positions[source]++];
          if (selected_variables.count (abs (literal)))
            continue;
          auto found = by_literal.find (literal);
          if (found == by_literal.end ())
            found = by_literal.find (-literal);
          if (found == by_literal.end ())
            continue;
          progress |= select_variable (result[found->second]);
          break;
        }
        if ((int) selected_variables.size () >= o.max_actions)
          break;
      }
      if (!progress)
        break;
    }
    for (const Proposal &proposal : result)
      if ((int) selected_variables.size () < o.max_actions)
        select_variable (proposal);
    return ordered;
  }

  if (o.max_actions <= 0 || (int) result.size () <= o.max_actions)
    return result;

  // Keep the native pair first, then fairly interleave heuristic sources.
  // A simple append-and-resize policy can otherwise consume the whole global
  // cap with activity literals and silently remove every JW/random control.
  std::vector<Proposal> ordered;
  ordered.reserve (o.max_actions);
  std::unordered_set<int> selected;
  const auto select = [&] (const Proposal &proposal) {
    if ((int) ordered.size () >= o.max_actions ||
        !selected.insert (proposal.external_literal).second)
      return false;
    ordered.push_back (proposal);
    return true;
  };
  for (const Proposal &proposal : result)
    if (proposal.sources & (PROPOSAL_NATIVE | PROPOSAL_NATIVE_OPPOSITE))
      select (proposal);

  const unsigned source_masks[] = {
      PROPOSAL_ACTIVITY_PHASE | PROPOSAL_ACTIVITY_OPPOSITE,
      PROPOSAL_JW,
      PROPOSAL_RANDOM,
  };
  size_t positions[] = {0, 0, 0};
  while ((int) ordered.size () < o.max_actions) {
    bool progress = false;
    for (size_t source = 0; source != 3; ++source) {
      while (positions[source] < result.size ()) {
        const Proposal &proposal = result[positions[source]++];
        if (!(proposal.sources & source_masks[source]) ||
            selected.count (proposal.external_literal))
          continue;
        progress |= select (proposal);
        break;
      }
      if ((int) ordered.size () >= o.max_actions)
        break;
    }
    if (!progress)
      break;
  }
  for (const Proposal &proposal : result)
    if ((int) ordered.size () < o.max_actions)
      select (proposal);
  return ordered;
}

static void emit_proposals (Internal *internal, int native_internal) {
  FILE *f = internal->decisiontrace_runtime.output;
  const std::vector<Proposal> actions =
      collect_proposals (internal, native_internal);
  fputs ("[\"pr\",", f);
  put_json_string (f, internal->decisiontrace_options.instance_id);
  fprintf (f, ",%lld,[", (long long)
           internal->decisiontrace_runtime.eligible_state_index);
  bool first = true;
  for (const Proposal &p : actions) {
    if (!first)
      fputc (',', f);
    first = false;
    fprintf (f, "[%d,%u,%.9g,%.9g,\"%016llx\"]",
             p.external_literal, p.sources, p.activity, p.jw,
             (unsigned long long) p.random_key);
  }
  fputs ("]]\n", f);
}

static int assignment_source (Internal *internal, int idx) {
  if (!internal->val (idx))
    return 0;
  const Var &v = internal->var (idx);
  if (!v.level)
    return 2;
  if (!v.reason)
    return 1;
  if (v.reason == internal->external_reason)
    return 5;
  return 2;
}

static void emit_int_vector_prefix (FILE *f, bool &first, int value) {
  if (!first)
    fputc (',', f);
  first = false;
  fprintf (f, "%d", value);
}

static bool emit_state (Internal *internal, int native_internal) {
  DecisionTraceRuntime &r = internal->decisiontrace_runtime;
  const DecisionTraceOptions &o = internal->decisiontrace_options;
  int64_t clauses = 0, occurrences = 0;
  int longest = 0;
  for (Clause *clause : internal->clauses) {
    if (!clause || clause->garbage)
      continue;
    ++clauses;
    occurrences += clause->size;
    longest = std::max (longest, (int) clause->size);
  }
  const int64_t nodes = 2 * (int64_t) internal->max_var + clauses;
  const int64_t estimated_payload_bytes =
      occurrences * 4 + (clauses + 1) * 8 +
      (int64_t) internal->max_var * 19 +
      (internal->external->max_var + 7) / 8;
  const char *failure = 0;
  if (!within_limit (nodes, o.max_state_nodes))
    failure = "state_nodes_exceeded";
  else if (!within_limit (occurrences, o.max_literal_occurrences))
    failure = "literal_occurrences_exceeded";
  else if (!within_limit (estimated_payload_bytes, o.max_snapshot_bytes))
    failure = "snapshot_bytes_exceeded";
  else if (o.max_clause_length > 0 && longest > o.max_clause_length)
    failure = "clause_length_exceeded";
  if (failure) {
    FILE *f = r.output;
    fputs ("[\"sf\",", f);
    put_json_string (f, o.instance_id);
    fprintf (f, ",%lld,", (long long) r.eligible_state_index);
    put_json_string (f, failure);
    fprintf (f, ",%lld,%lld,%d,%lld]\n", (long long) nodes,
             (long long) occurrences, longest,
             (long long) estimated_payload_bytes);
    ++r.materialization_failures;
    return false;
  }

  const StateFingerprint fingerprint = state_fingerprint (internal);
  FILE *f = r.output;
  // Keep a small, independently parseable locator in front of the potentially
  // very large state record.  The Python orchestrator can validate replay and
  // launch EVAL actions without materializing the JSON state arrays in RAM.
  fputs ("[\"sl\",", f);
  put_json_string (f, o.instance_id);
  fprintf (f, ",%lld,%lld,%lld,%d,%zu,%lld,%lld,%lld,%lld,",
           (long long) r.eligible_state_index,
           (long long) r.callback_attempt_index,
           (long long) r.raw_decision_index, r.pending_level,
           (size_t) r.pending_trail_size, (long long) r.pending_conflicts,
           (long long) r.pending_decisions,
           (long long) r.pending_propagations,
           (long long) r.pending_restarts);
  put_hash (f, r.pending_prefix_hash);
  fprintf (f, ",%d,", external_literal (internal, native_internal));
  put_hash (f, fingerprint.assignment);
  fputc (',', f);
  put_hash (f, fingerprint.trail);
  fputc (',', f);
  put_hash (f, fingerprint.clause_canonical);
  fputc (',', f);
  put_hash (f, fingerprint.clause_order);
  fputc (',', f);
  put_hash (f, fingerprint.eligibility);
  fprintf (f, ",%lld,%lld,%d,%lld]\n", (long long) nodes,
           (long long) occurrences, longest,
           (long long) estimated_payload_bytes);

  fputs ("[\"st\",", f);
  put_json_string (f, o.instance_id);
  fprintf (f, ",%lld,%lld,%lld,%d,%zu,%lld,%lld,%lld,%lld,",
           (long long) r.eligible_state_index,
           (long long) r.callback_attempt_index,
           (long long) r.raw_decision_index, r.pending_level,
           (size_t) r.pending_trail_size, (long long) r.pending_conflicts,
           (long long) r.pending_decisions,
           (long long) r.pending_propagations,
           (long long) r.pending_restarts);
  put_hash (f, r.pending_prefix_hash);
  fprintf (f, ",%d,%d,%d,%lld,%lld,",
           external_literal (internal, native_internal), internal->max_var,
           internal->external->max_var,
           (long long) clauses, (long long) occurrences);
  put_hash (f, fingerprint.assignment);
  fputc (',', f);
  put_hash (f, fingerprint.trail);
  fputc (',', f);
  put_hash (f, fingerprint.clause_canonical);
  fputc (',', f);
  put_hash (f, fingerprint.clause_order);
  fputc (',', f);
  put_hash (f, fingerprint.eligibility);
  fputs (",[", f);

  for (int idx = 1; idx <= internal->max_var; ++idx) {
    if (idx > 1)
      fputc (',', f);
    fprintf (f, "%d", internal->externalize (idx));
  }
  fputs ("],\"", f);
  unsigned byte = 0, bit = 0;
  static const char hex[] = "0123456789abcdef";
  for (int eidx = 1; eidx <= internal->external->max_var; ++eidx) {
    const int ilit = existing_internal_literal (internal, eidx);
    if (ilit && candidate_internal_variable (internal, abs (ilit)))
      byte |= 1u << bit;
    if (++bit == 8 || eidx == internal->external->max_var) {
      fputc (hex[(byte >> 4) & 15], f);
      fputc (hex[byte & 15], f);
      byte = bit = 0;
    }
  }
  fputs ("\",[", f);

  bool first = true;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    emit_int_vector_prefix (f, first, (int) internal->val (idx));
  fputs ("],[", f);
  first = true;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    emit_int_vector_prefix (f, first,
                            internal->val (idx) ? internal->var (idx).level : -1);
  fputs ("],[", f);
  first = true;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    emit_int_vector_prefix (f, first, assignment_source (internal, idx));
  fputs ("],[", f);
  first = true;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    emit_int_vector_prefix (
        f, first, internal->val (idx) ? internal->var (idx).trail : -1);
  fputs ("],[", f);
  first = true;
  const double max_float = std::numeric_limits<float>::max ();
  for (int idx = 1; idx <= internal->max_var; ++idx) {
    if (!first)
      fputc (',', f);
    first = false;
    double value = internal->score (idx);
    if (!std::isfinite (value))
      value = 0;
    value = std::max (-max_float, std::min (max_float, value));
    fprintf (f, "%.9g", (float) value);
  }
  fputs ("],[", f);
  first = true;
  for (int idx = 1; idx <= internal->max_var; ++idx)
    emit_int_vector_prefix (f, first, (int) internal->phases.saved[idx]);

  fputs ("],[0", f);
  int64_t offset = 0;
  for (Clause *clause : internal->clauses) {
    if (!clause || clause->garbage)
      continue;
    offset += clause->size;
    fprintf (f, ",%lld", (long long) offset);
  }
  fputs ("],[", f);
  first = true;
  for (Clause *clause : internal->clauses) {
    if (!clause || clause->garbage)
      continue;
    for (const int lit : *clause)
      emit_int_vector_prefix (f, first, lit);
  }
  fputs ("]]\n", f);
  ++r.materialized_states;
  return true;
}

static void update_prefix_hash (Internal *internal, int applied_internal) {
  DecisionTraceRuntime &r = internal->decisiontrace_runtime;
  hash_word (r.prefix_hash,
             (uint64_t) (int64_t) external_literal (internal, applied_internal));
  hash_word (r.prefix_hash, (uint64_t) r.pending_level);
  hash_word (r.prefix_hash, (uint64_t) r.pending_conflicts);
  hash_word (r.prefix_hash, (uint64_t) r.pending_decisions);
  hash_word (r.prefix_hash, (uint64_t) r.pending_restarts);
}

} // namespace

void Internal::configure_decisiontrace (const DecisionTraceOptions &options) {
  if (decisiontrace_runtime.started && !decisiontrace_runtime.finished)
    error ("can not reconfigure DecisionTrace during a run");
  decisiontrace_options = options;
}

void External::configure_decisiontrace (const DecisionTraceOptions &options) {
  internal->configure_decisiontrace (options);
}

void Solver::configure_decisiontrace (const DecisionTraceOptions &options) {
  REQUIRE_VALID_OR_SOLVING_STATE ();
  external->configure_decisiontrace (options);
}

int External::decisiontrace_callback_action () {
  return internal->decisiontrace_callback_action ();
}

int Solver::decisiontrace_callback_action () {
  return external->decisiontrace_callback_action ();
}

void Internal::decisiontrace_begin_search () {
  if (!decisiontrace_options.enabled || decisiontrace_runtime.started)
    return;
  decisiontrace_runtime = DecisionTraceRuntime ();
  for (const int64_t target : decisiontrace_options.target_eligible_indices)
    if (target > 0)
      decisiontrace_runtime.capture_targets.insert (target);
  if (decisiontrace_options.output_path == "-") {
    decisiontrace_runtime.output = stdout;
    decisiontrace_runtime.owns_output = false;
  } else {
    decisiontrace_runtime.output =
        fopen (decisiontrace_options.output_path.c_str (), "wb");
    if (!decisiontrace_runtime.output)
      error ("can not open DecisionTrace output '%s'",
             decisiontrace_options.output_path.c_str ());
    decisiontrace_runtime.owns_output = true;
  }
  decisiontrace_runtime.started = true;
  if (decisiontrace_options.emit_schema)
    write_schema (this);
}

int Internal::decisiontrace_callback_action () {
  if (!decisiontrace_options.enabled || !decisiontrace_runtime.started)
    return 0;
  DecisionTraceRuntime &r = decisiontrace_runtime;
  const DecisionTraceOptions &o = decisiontrace_options;
  if (r.callback_pending)
    error ("DecisionTrace callback without committed previous decision");
  ++r.callback_attempt_index;
  ++r.raw_decision_index;
  r.callback_pending = true;
  r.pending_level = level;
  r.pending_trail_size = trail.size ();
  r.pending_conflicts = stats.conflicts;
  r.pending_decisions = stats.decisions;
  r.pending_propagations = stats.propagations.search;
  r.pending_restarts = stats.restarts;
  r.pending_prefix_hash = r.prefix_hash;

  const int candidates = count_candidates (this);
  int64_t state_nodes = 0;
  r.pending_eligible = cheap_eligible (this, candidates, state_nodes);
  r.pending_candidate_count = candidates;
  r.pending_state_nodes = state_nodes;
  if (r.pending_eligible)
    ++r.eligible_state_index;

  if (o.mode == "eval" && r.pending_eligible && !r.eval_target_reached &&
      r.eligible_state_index == o.eval_target_eligible_index) {
    r.eval_target_reached = true;
    r.eval_start_conflicts = stats.conflicts;
    r.eval_start_decisions = stats.decisions;
    r.eval_start_propagations = stats.propagations.search;
    r.eval_start_restarts = stats.restarts;
    r.eval_callback_index = r.callback_attempt_index;
    r.eval_raw_decision_index = r.raw_decision_index;
    r.eval_prefix_hash = r.prefix_hash;

    bool locator_match = true;
    if (o.expected_callback_index > 0)
      locator_match &= r.callback_attempt_index == o.expected_callback_index;
    if (o.expected_raw_decision_index > 0)
      locator_match &= r.raw_decision_index == o.expected_raw_decision_index;
    if (o.expected_prefix_hash)
      locator_match &= r.prefix_hash == o.expected_prefix_hash;
    if (o.expected_conflicts >= 0)
      locator_match &= stats.conflicts == o.expected_conflicts;
    if (o.expected_decisions >= 0)
      locator_match &= stats.decisions == o.expected_decisions;
    if (o.expected_propagations >= 0)
      locator_match &=
          stats.propagations.search == o.expected_propagations;
    if (o.expected_restarts >= 0)
      locator_match &= stats.restarts == o.expected_restarts;
    if (o.expected_trail_size >= 0)
      locator_match &=
          (int64_t) trail.size () == o.expected_trail_size;
    const bool need_full_fingerprint =
        o.expected_assignment_hash || o.expected_trail_hash ||
        o.expected_clause_canonical_hash || o.expected_clause_order_hash ||
        o.expected_eligibility_hash;
    if (locator_match && need_full_fingerprint) {
      const StateFingerprint fingerprint = state_fingerprint (this);
      if (o.expected_assignment_hash)
        locator_match &=
            fingerprint.assignment == o.expected_assignment_hash;
      if (o.expected_trail_hash)
        locator_match &= fingerprint.trail == o.expected_trail_hash;
      if (o.expected_clause_canonical_hash)
        locator_match &= fingerprint.clause_canonical ==
                         o.expected_clause_canonical_hash;
      if (o.expected_clause_order_hash)
        locator_match &=
            fingerprint.clause_order == o.expected_clause_order_hash;
      if (o.expected_eligibility_hash)
        locator_match &=
            fingerprint.eligibility == o.expected_eligibility_hash;
    }
    r.eval_locator_match = locator_match;
    r.eval_native_external_literal = o.expected_native_external_literal;

    if (o.eval_forced_external_literal) {
      const int forced =
          existing_internal_literal (this, o.eval_forced_external_literal);
      r.eval_force_eligible =
          forced && candidate_internal_variable (this, abs (forced));
      if (!locator_match || !r.eval_force_eligible) {
        r.stop_requested = true;
        return 0;
      }
      r.pending_requested_external_literal =
          o.eval_forced_external_literal;
      ++r.forced_actions;
      return o.eval_forced_external_literal;
    }
    r.eval_force_eligible = locator_match;
    if (!locator_match)
      r.stop_requested = true;
  }
  return 0;
}

void Internal::decisiontrace_commit_decision (int applied_internal) {
  if (!decisiontrace_options.enabled || !decisiontrace_runtime.started)
    return;
  DecisionTraceRuntime &r = decisiontrace_runtime;
  const DecisionTraceOptions &o = decisiontrace_options;
  if (!r.callback_pending)
    return; // private solver steps do not correspond to online cb_decide

  if (o.mode == "discover" && r.pending_eligible && prefix_budget_ok (this)) {
    const bool in_locator_window =
        r.eligible_state_index <= o.discover_eligible_window_end;
    const bool after_restart =
        r.pending_restarts > 0 &&
        r.emitted_post_restart_state_count <
            o.discover_post_restart_keep_count;
    if (in_locator_window || after_restart) {
      emit_trace_record (this, r.pending_candidate_count,
                         r.pending_state_nodes, applied_internal);
      if (in_locator_window)
        ++r.emitted_window_states;
      if (after_restart)
        ++r.emitted_post_restart_state_count;
    }
    const bool locator_window_complete =
        !o.discover_eligible_window_end ||
        r.eligible_state_index >= o.discover_eligible_window_end;
    const bool restart_complete =
        o.discover_post_restart_keep_count <= 0 ||
        r.emitted_post_restart_state_count >=
            o.discover_post_restart_keep_count;
    if (locator_window_complete && restart_complete)
      r.stop_requested = true;
  } else if (o.mode == "capture" && r.pending_eligible) {
    if (r.capture_targets.count (r.eligible_state_index)) {
      if (emit_state (this, applied_internal))
        emit_proposals (this, applied_internal);
      r.captured_targets.insert (r.eligible_state_index);
      if (r.captured_targets.size () == r.capture_targets.size ())
        r.stop_requested = true;
    }
  } else if (o.mode == "eval" && r.eval_target_reached &&
             r.eligible_state_index == o.eval_target_eligible_index &&
             !r.eval_force_applied) {
    const int applied_external = external_literal (this, applied_internal);
    r.eval_applied_internal_literal = applied_internal;
    r.eval_applied_external_literal = applied_external;
    if (!o.eval_forced_external_literal) {
      r.eval_native_internal_literal = applied_internal;
      r.eval_native_external_literal = applied_external;
      if (o.expected_native_external_literal &&
          applied_external != o.expected_native_external_literal) {
        r.eval_locator_match = false;
        r.stop_requested = true;
      }
    }
    r.eval_force_applied =
        r.eval_locator_match && r.eval_force_eligible &&
        (!o.eval_forced_external_literal ||
         applied_external == o.eval_forced_external_literal);
    if (r.eval_force_applied && o.eval_max_timeout > 0)
      r.eval_timeout_deadline = absolute_real_time () + o.eval_max_timeout;
  }

  update_prefix_hash (this, applied_internal);
  r.callback_pending = false;
}

bool Internal::decisiontrace_timeout_expired () {
  if (!decisiontrace_options.enabled || !decisiontrace_runtime.started)
    return false;
  DecisionTraceRuntime &r = decisiontrace_runtime;
  if (decisiontrace_options.mode != "eval" || !r.eval_force_applied ||
      r.stop_requested || r.eval_timeout_deadline <= 0 ||
      absolute_real_time () < r.eval_timeout_deadline)
    return false;
  r.stopped_by_horizon = true;
  r.stop_requested = true;
  return true;
}

bool Internal::decisiontrace_should_stop () {
  if (!decisiontrace_options.enabled || !decisiontrace_runtime.started)
    return false;
  DecisionTraceRuntime &r = decisiontrace_runtime;
  if (r.stop_requested)
    return true;
  if (decisiontrace_timeout_expired ())
    return true;
  // Prefix limits are hard replay limits, not merely output filters.  This is
  // particularly important for large SATCompetition instances: a missing
  // late target must not make DISCOVER/CAPTURE/EVAL solve indefinitely.
  if ((decisiontrace_options.mode != "eval" || !r.eval_target_reached) &&
      !prefix_budget_has_room (this)) {
    r.stop_requested = true;
    return true;
  }
  if (decisiontrace_options.mode != "eval" || !r.eval_target_reached)
    return false;
  // Give the chosen action at least its immediate Boolean propagation.  In
  // particular, a one-decision horizon must not stop directly after merely
  // putting the literal on the trail.
  if (propagated < trail.size ())
    return false;
  const int64_t dc = stats.conflicts - r.eval_start_conflicts;
  const int64_t dd = stats.decisions - r.eval_start_decisions;
  const int64_t dp = stats.propagations.search - r.eval_start_propagations;
  const DecisionTraceOptions &o = decisiontrace_options;
  if ((o.eval_max_conflicts && dc >= o.eval_max_conflicts) ||
      (o.eval_max_decisions && dd >= o.eval_max_decisions) ||
      (o.eval_max_propagations && dp >= o.eval_max_propagations)) {
    r.stopped_by_horizon = true;
    r.stop_requested = true;
    return true;
  }
  return false;
}

void Internal::decisiontrace_finish (int status) {
  if (!decisiontrace_options.enabled || !decisiontrace_runtime.started ||
      decisiontrace_runtime.finished)
    return;
  DecisionTraceRuntime &r = decisiontrace_runtime;
  FILE *f = r.output;

  if (decisiontrace_options.mode == "eval") {
    const int64_t dc = r.eval_target_reached
                           ? stats.conflicts - r.eval_start_conflicts
                           : 0;
    const int64_t dd = r.eval_target_reached
                           ? stats.decisions - r.eval_start_decisions
                           : 0;
    const int64_t dp = r.eval_target_reached
                           ? stats.propagations.search - r.eval_start_propagations
                           : 0;
    const int64_t dr = r.eval_target_reached
                           ? stats.restarts - r.eval_start_restarts
                           : 0;
    const bool censored = status == 0 || !r.eval_target_reached ||
                          !r.eval_force_applied || r.stopped_by_horizon;
    fputs ("[\"ev\",", f);
    put_json_string (f, decisiontrace_options.instance_id);
    fprintf (f, ",%lld,%lld,%lld,%d,%s,%s,%s,%s,",
             (long long) decisiontrace_options.eval_target_eligible_index,
             (long long) r.eval_callback_index,
             (long long) r.eval_raw_decision_index,
             decisiontrace_options.eval_forced_external_literal,
             r.eval_target_reached ? "true" : "false",
             r.eval_locator_match ? "true" : "false",
             r.eval_force_eligible ? "true" : "false",
             r.eval_force_applied ? "true" : "false");
    put_hash (f, r.eval_prefix_hash);
    fprintf (f, ",%d,%d,%d,%s,%s,%lld,%lld,%lld,%lld]\n",
             r.eval_native_external_literal, r.eval_applied_external_literal,
             status, censored ? "true" : "false",
             r.stopped_by_horizon ? "true" : "false", (long long) dc,
             (long long) dd, (long long) dp, (long long) dr);
  }

  fputs ("[\"out\",", f);
  put_json_string (f, decisiontrace_options.instance_id);
  fputc (',', f);
  put_json_string (f, decisiontrace_options.mode);
  fprintf (f, ",%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld]\n", status,
           (long long) stats.conflicts, (long long) stats.decisions,
           (long long) stats.propagations.search,
           (long long) stats.restarts,
           (long long) r.callback_attempt_index,
           (long long) r.eligible_state_index,
           (long long) r.materialized_states,
           (long long) r.materialization_failures);
  if (fflush (f) || ferror (f))
    error ("failed writing DecisionTrace output '%s'",
           decisiontrace_options.output_path.c_str ());
  if (r.owns_output && fclose (f))
    error ("failed closing DecisionTrace output '%s'",
           decisiontrace_options.output_path.c_str ());
  r.output = 0;
  r.owns_output = false;
  r.finished = true;
}

} // namespace CaDiCaL
