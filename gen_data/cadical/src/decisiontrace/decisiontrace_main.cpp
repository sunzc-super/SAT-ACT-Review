#include "cadical.hpp"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using CaDiCaL::DecisionTraceOptions;
using CaDiCaL::ExternalPropagator;
using CaDiCaL::Solver;

namespace {

class OfflineDecisionPropagator : public ExternalPropagator {
  Solver *solver;

public:
  explicit OfflineDecisionPropagator (Solver *solver_arg)
      : solver (solver_arg) {
    is_lazy = false;
  }

  void notify_assignment (const std::vector<int> &) override {}
  void notify_new_decision_level () override {}
  void notify_backtrack (size_t) override {}
  bool cb_check_found_model (const std::vector<int> &) override {
    return true;
  }
  int cb_decide () override { return solver->decisiontrace_callback_action (); }
  bool cb_has_external_clause (bool &is_forgettable) override {
    is_forgettable = false;
    return false;
  }
  int cb_add_external_clause_lit () override { return 0; }
};

static void usage (const char *program) {
  std::cout
      << "usage: " << program << " [options] <input.cnf[.gz|.xz]>\n\n"
      << "Offline DecisionTrace-ActionEval worker. Directory traversal and "
         "part assembly belong to the Python runner.\n\n"
      << "Core options:\n"
      << "  --mode=discover|capture|eval\n"
      << "  --output=<path|->\n"
      << "  --instance-id=<stable-id>\n"
      << "  --targets=1,2,5              capture mode\n"
      << "  --target=5 --force-literal=-17  eval mode; zero is native\n"
      << "  --expected-prefix-hash=<hex> and --expected-*=N verify replay\n"
      << "  --no-schema                  omit per-process schema header\n\n"
      << "Eligibility and target-search budgets:\n"
      << "  --max-state-nodes=N\n"
      << "  --max-literal-occurrences=N\n"
      << "  --max-snapshot-bytes=N\n"
      << "  --max-clause-length=N        zero disables\n"
      << "  --max-prefix-conflicts=N\n"
      << "  --max-prefix-decisions=N\n"
      << "  --max-prefix-propagations=N\n"
      << "  --max-prefix-callbacks=N\n"
      << "  --discover-eligible-window-end=N\n"
      << "  --discover-post-restart-keep-count=N\n\n"
      << "Evaluation horizons (zero means complete solve):\n"
      << "  --eval-max-conflicts=N --eval-max-decisions=N\n"
      << "  --eval-max-propagations=N\n"
      << "  --eval-max-timeout=SECONDS\n\n"
      << "Candidate proposal controls:\n"
      << "  --activity-top-k=N --jw-top-k=N --random-top-k=N\n"
      << "  --max-actions-mode=default|variable\n"
      << "  --max-actions=N              signed literals or variables; 0 unlimited\n"
      << "  --seed=N\n\n"
      << "Solver controls:\n"
      << "  --plain                           disable preprocessing\n"
      << "  --set=<cadical-option>=<integer>  repeatable\n";
}

static bool split_option (const std::string &arg, const char *name,
                          std::string &value) {
  const std::string prefix = std::string ("--") + name + "=";
  if (arg.compare (0, prefix.size (), prefix))
    return false;
  value = arg.substr (prefix.size ());
  return true;
}

static bool parse_i64 (const std::string &text, int64_t &value) {
  if (text.empty ())
    return false;
  errno = 0;
  char *end = 0;
  const long long parsed = strtoll (text.c_str (), &end, 10);
  if (errno || !end || *end)
    return false;
  value = (int64_t) parsed;
  return true;
}

static bool parse_u64 (const std::string &text, uint64_t &value) {
  if (text.empty () || text[0] == '-')
    return false;
  errno = 0;
  char *end = 0;
  const unsigned long long parsed = strtoull (text.c_str (), &end, 10);
  if (errno || !end || *end)
    return false;
  value = (uint64_t) parsed;
  return true;
}

static bool parse_hex_u64 (const std::string &text, uint64_t &value) {
  if (text.empty ())
    return false;
  errno = 0;
  char *end = 0;
  const unsigned long long parsed = strtoull (text.c_str (), &end, 16);
  if (errno || !end || *end)
    return false;
  value = (uint64_t) parsed;
  return true;
}

static bool parse_int (const std::string &text, int &value) {
  int64_t parsed = 0;
  if (!parse_i64 (text, parsed) || parsed < INT_MIN || parsed > INT_MAX)
    return false;
  value = (int) parsed;
  return true;
}

static bool parse_nonnegative_i64 (const std::string &text, int64_t &value) {
  return parse_i64 (text, value) && value >= 0;
}

static bool parse_nonnegative_int (const std::string &text, int &value) {
  return parse_int (text, value) && value >= 0;
}

static bool parse_nonnegative_double (const std::string &text,
                                      double &value) {
  if (text.empty ())
    return false;
  errno = 0;
  char *end = 0;
  const double parsed = strtod (text.c_str (), &end);
  if (errno || !end || *end || !std::isfinite (parsed) || parsed < 0)
    return false;
  value = parsed;
  return true;
}

static bool parse_targets (const std::string &text,
                           std::vector<int64_t> &targets) {
  std::set<int64_t> unique;
  std::stringstream stream (text);
  std::string item;
  while (std::getline (stream, item, ',')) {
    int64_t value = 0;
    if (!parse_i64 (item, value) || value <= 0)
      return false;
    unique.insert (value);
  }
  if (unique.empty ())
    return false;
  targets.assign (unique.begin (), unique.end ());
  return true;
}

static bool parse_solver_setting (const std::string &text,
                                  std::pair<std::string, int> &setting) {
  const size_t equal = text.find ('=');
  if (!equal || equal == std::string::npos)
    return false;
  int value = 0;
  if (!parse_int (text.substr (equal + 1), value))
    return false;
  setting = std::make_pair (text.substr (0, equal), value);
  return !setting.first.empty ();
}

} // namespace

int main (int argc, char **argv) {
  DecisionTraceOptions options;
  options.enabled = true;
  bool plain = false;
  std::vector<std::pair<std::string, int>> solver_settings;
  std::string input;

  for (int i = 1; i < argc; ++i) {
    const std::string arg (argv[i]);
    std::string value;
    if (arg == "-h" || arg == "--help") {
      usage (argv[0]);
      return 0;
    } else if (arg == "--plain") {
      plain = true;
    } else if (arg == "--no-schema") {
      options.emit_schema = false;
    } else if (split_option (arg, "mode", value)) {
      options.mode = value;
    } else if (split_option (arg, "output", value)) {
      options.output_path = value;
    } else if (split_option (arg, "instance-id", value)) {
      options.instance_id = value;
    } else if (split_option (arg, "targets", value)) {
      if (!parse_targets (value, options.target_eligible_indices)) {
        std::cerr << "invalid --targets: " << value << '\n';
        return 2;
      }
    } else if (split_option (arg, "target", value)) {
      if (!parse_i64 (value, options.eval_target_eligible_index) ||
          options.eval_target_eligible_index <= 0) {
        std::cerr << "invalid --target: " << value << '\n';
        return 2;
      }
    } else if (split_option (arg, "force-literal", value)) {
      if (!parse_int (value, options.eval_forced_external_literal)) {
        std::cerr << "invalid --force-literal: " << value << '\n';
        return 2;
      }
    } else if (split_option (arg, "expected-callback", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_callback_index))
        return 2;
    } else if (split_option (arg, "expected-raw-decision", value)) {
      if (!parse_nonnegative_i64 (value,
                                  options.expected_raw_decision_index))
        return 2;
    } else if (split_option (arg, "expected-prefix-hash", value)) {
      if (!parse_hex_u64 (value, options.expected_prefix_hash))
        return 2;
    } else if (split_option (arg, "expected-assignment-hash", value)) {
      if (!parse_hex_u64 (value, options.expected_assignment_hash))
        return 2;
    } else if (split_option (arg, "expected-trail-hash", value)) {
      if (!parse_hex_u64 (value, options.expected_trail_hash))
        return 2;
    } else if (split_option (arg, "expected-clause-canonical-hash", value)) {
      if (!parse_hex_u64 (value,
                          options.expected_clause_canonical_hash))
        return 2;
    } else if (split_option (arg, "expected-clause-order-hash", value)) {
      if (!parse_hex_u64 (value, options.expected_clause_order_hash))
        return 2;
    } else if (split_option (arg, "expected-eligibility-hash", value)) {
      if (!parse_hex_u64 (value, options.expected_eligibility_hash))
        return 2;
    } else if (split_option (arg, "expected-conflicts", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_conflicts))
        return 2;
    } else if (split_option (arg, "expected-decisions", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_decisions))
        return 2;
    } else if (split_option (arg, "expected-propagations", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_propagations))
        return 2;
    } else if (split_option (arg, "expected-restarts", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_restarts))
        return 2;
    } else if (split_option (arg, "expected-trail", value)) {
      if (!parse_nonnegative_i64 (value, options.expected_trail_size))
        return 2;
    } else if (split_option (arg, "expected-native-literal", value)) {
      if (!parse_int (value, options.expected_native_external_literal))
        return 2;
    } else if (split_option (arg, "max-state-nodes", value)) {
      if (!parse_nonnegative_i64 (value, options.max_state_nodes))
        return 2;
    } else if (split_option (arg, "max-literal-occurrences", value)) {
      if (!parse_nonnegative_i64 (value, options.max_literal_occurrences))
        return 2;
    } else if (split_option (arg, "max-snapshot-bytes", value)) {
      if (!parse_nonnegative_i64 (value, options.max_snapshot_bytes))
        return 2;
    } else if (split_option (arg, "max-clause-length", value)) {
      if (!parse_nonnegative_int (value, options.max_clause_length))
        return 2;
    } else if (split_option (arg, "max-prefix-conflicts", value)) {
      if (!parse_nonnegative_i64 (value, options.max_prefix_conflicts))
        return 2;
    } else if (split_option (arg, "max-prefix-decisions", value)) {
      if (!parse_nonnegative_i64 (value, options.max_prefix_decisions))
        return 2;
    } else if (split_option (arg, "max-prefix-propagations", value)) {
      if (!parse_nonnegative_i64 (value, options.max_prefix_propagations))
        return 2;
    } else if (split_option (arg, "max-prefix-callbacks", value)) {
      if (!parse_nonnegative_i64 (value, options.max_prefix_callbacks))
        return 2;
    } else if (split_option (arg, "discover-eligible-window-end", value)) {
      if (!parse_nonnegative_i64 (value,
                                  options.discover_eligible_window_end))
        return 2;
    } else if (split_option (arg, "discover-post-restart-keep-count", value)) {
      if (!parse_nonnegative_int (value,
                                  options.discover_post_restart_keep_count))
        return 2;
    } else if (split_option (arg, "eval-max-conflicts", value)) {
      if (!parse_nonnegative_i64 (value, options.eval_max_conflicts))
        return 2;
    } else if (split_option (arg, "eval-max-decisions", value)) {
      if (!parse_nonnegative_i64 (value, options.eval_max_decisions))
        return 2;
    } else if (split_option (arg, "eval-max-propagations", value)) {
      if (!parse_nonnegative_i64 (value, options.eval_max_propagations))
        return 2;
    } else if (split_option (arg, "eval-max-timeout", value)) {
      if (!parse_nonnegative_double (value, options.eval_max_timeout)) {
        std::cerr << "invalid --eval-max-timeout: " << value << '\n';
        return 2;
      }
    } else if (split_option (arg, "activity-top-k", value)) {
      if (!parse_nonnegative_int (value, options.activity_top_k))
        return 2;
    } else if (split_option (arg, "jw-top-k", value)) {
      if (!parse_nonnegative_int (value, options.jw_top_k))
        return 2;
    } else if (split_option (arg, "random-top-k", value)) {
      if (!parse_nonnegative_int (value, options.random_top_k))
        return 2;
    } else if (split_option (arg, "max-actions", value)) {
      if (!parse_nonnegative_int (value, options.max_actions))
        return 2;
    } else if (split_option (arg, "max-actions-mode", value)) {
      options.max_actions_mode = value;
    } else if (split_option (arg, "seed", value)) {
      if (!parse_u64 (value, options.seed))
        return 2;
    } else if (split_option (arg, "set", value)) {
      std::pair<std::string, int> setting;
      if (!parse_solver_setting (value, setting)) {
        std::cerr << "invalid --set: " << value << '\n';
        return 2;
      }
      solver_settings.push_back (setting);
    } else if (!arg.empty () && arg[0] == '-') {
      std::cerr << "unknown option: " << arg << '\n';
      return 2;
    } else if (!input.empty ()) {
      std::cerr << "multiple input paths: " << input << " and " << arg
                << '\n';
      return 2;
    } else
      input = arg;
  }

  if (input.empty ()) {
    usage (argv[0]);
    return 2;
  }
  if (options.mode != "discover" && options.mode != "capture" &&
      options.mode != "eval") {
    std::cerr << "invalid --mode: " << options.mode << '\n';
    return 2;
  }
  if (options.max_actions_mode != "default" &&
      options.max_actions_mode != "variable") {
    std::cerr << "invalid --max-actions-mode: " << options.max_actions_mode
              << '\n';
    return 2;
  }
  if (options.mode == "capture" &&
      options.target_eligible_indices.empty ()) {
    std::cerr << "capture mode requires --targets\n";
    return 2;
  }
  if (options.mode == "eval" &&
      options.eval_target_eligible_index <= 0) {
    std::cerr << "eval mode requires --target\n";
    return 2;
  }

  Solver solver;
  if (plain && !solver.configure ("plain")) {
    std::cerr << "failed to configure plain mode\n";
    return 2;
  }
  solver.set ("quiet", 1);
  solver.set ("seed", (int) (options.seed % (uint64_t) INT_MAX));
  for (const auto &setting : solver_settings) {
    if (setting.first == "quiet" && setting.second != 1) {
      std::cerr << "DecisionTrace requires --set=quiet=1 so solver text "
                   "does not corrupt JSONL output\n";
      return 2;
    }
    if (!solver.set (setting.first.c_str (), setting.second)) {
      std::cerr << "unknown or invalid solver setting: " << setting.first
                << '=' << setting.second << '\n';
      return 2;
    }
  }
  solver.configure_decisiontrace (options);

  int variables = 0;
  const char *error = solver.read_dimacs (input.c_str (), variables, 1);
  if (error) {
    std::cerr << input << ": " << error << '\n';
    return 2;
  }
  OfflineDecisionPropagator propagator (&solver);
  solver.connect_external_propagator (&propagator);
  for (int variable = 1; variable <= variables; ++variable)
    solver.add_observed_var (variable);
  solver.solve ();
  solver.disconnect_external_propagator ();
  return 0;
}
