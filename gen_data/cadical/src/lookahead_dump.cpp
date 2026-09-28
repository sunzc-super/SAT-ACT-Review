#include "internal.hpp"
#include "cadical.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace CaDiCaL {

namespace {

enum LookaheadDumpProfile {
  LOOKAHEAD_DUMP_FULL = 0,
  LOOKAHEAD_DUMP_COMPACT = 1,
  LOOKAHEAD_DUMP_LITE = 2,
  LOOKAHEAD_DUMP_STATS = 3,
  LOOKAHEAD_DUMP_TABULAR = 4
};

enum LookaheadDumpMode {
  LOOKAHEAD_DUMP_LEGACY = 0,
  LOOKAHEAD_DUMP_V2_BASIC = 1,
  LOOKAHEAD_DUMP_V2_CUBE = 2,
  LOOKAHEAD_DUMP_V2_EVAL = 3,
  LOOKAHEAD_DUMP_V2_FULL = 4,
  LOOKAHEAD_DUMP_V2_EVAL_ONLY = 5
};

struct LiteralEffects {
  int satisfied_clause_count = 0;
  int shortened_clause_count = 0;
  int new_binary_clause_count = 0;
  int new_unit_clause_count = 0;
};

struct ProbeRecord {
  int lit = 0;
  int probe_index = 0;
  size_t trail_before = 0;
  size_t trail_after = 0;
  int hbrs = 0;
  int extra_implied_count = 0;
  LiteralEffects effects;
};

struct DecisionEvalCandidate {
  int lit = 0;
  std::vector<std::string> source;
};

struct LiteralScore {
  int lit = 0;
  double score = 0;
};

// Return whether the selected dump mode uses the v2 schema extensions.
// 返回所选 dump mode 是否使用 v2 schema 扩展.
static bool is_v2_mode (int mode) { return mode != LOOKAHEAD_DUMP_LEGACY; }

// Return whether the selected dump mode includes first-decision evaluation.
// 返回所选 dump mode 是否包含第一次决策评估.
static bool mode_has_eval (int mode) {
  return mode == LOOKAHEAD_DUMP_V2_EVAL ||
         mode == LOOKAHEAD_DUMP_V2_FULL ||
         mode == LOOKAHEAD_DUMP_V2_EVAL_ONLY;
}

// Return whether the selected dump mode can trigger cube fallback.
// 返回所选 dump mode 是否可以触发 cube fallback.
static bool mode_has_cube (int mode) {
  return mode == LOOKAHEAD_DUMP_V2_CUBE || mode == LOOKAHEAD_DUMP_V2_FULL;
}

// Return whether the selected dump mode skips lookahead probing.
// 返回所选 dump mode 是否跳过 lookahead probing.
static bool mode_eval_only (int mode) {
  return mode == LOOKAHEAD_DUMP_V2_EVAL_ONLY;
}

// Return whether rows use compact arrays with full event names.
// 返回 row 是否使用带完整事件名的 compact array 格式.
static bool is_tabular_profile (int profile) {
  return profile == LOOKAHEAD_DUMP_TABULAR;
}

// Return the event label for compact-like row profiles.
// 返回 compact-like row profile 使用的事件标签.
static const char *row_label (int profile, const char *short_name,
                              const char *full_name) {
  return is_tabular_profile (profile) ? full_name : short_name;
}

// Convert a literal into the file-name-safe P7 or N7 spelling.
// 将 literal 转换为文件名安全的 P7 或 N7 形式.
static std::string lit_tag (int lit) {
  char buffer[64];
  snprintf (buffer, sizeof buffer, "%c%d", lit < 0 ? 'N' : 'P', abs (lit));
  return buffer;
}

// Insert a cube literal tag before the lookahead JSONL suffix.
// 在 lookahead JSONL 后缀前插入 cube literal 标记.
static std::string cube_child_path (const std::string &path, int lit) {
  const std::string suffix = ".lookahead.jsonl";
  const size_t pos = path.rfind (suffix);
  if (pos == std::string::npos)
    return path + ".cube." + lit_tag (lit) + ".jsonl";
  return path.substr (0, pos) + ".cube." + lit_tag (lit) + suffix;
}

// Dump one clause literal array as a JSON array.
// 将一个子句的 literal 数组写成 JSON 数组.
static void dump_lits (FILE *file, const Clause *c) {
  fputc ('[', file);
  for (int i = 0; i < c->size; i++) {
    if (i)
      fputc (',', file);
    fprintf (file, "%d", c->literals[i]);
  }
  fputc (']', file);
}

// Dump an integer vector as a JSON array.
// 将一个整数向量写成 JSON 数组.
static void dump_int_vector (FILE *file, const std::vector<int> &lits) {
  fputc ('[', file);
  for (size_t i = 0; i < lits.size (); i++) {
    if (i)
      fputc (',', file);
    fprintf (file, "%d", lits[i]);
  }
  fputc (']', file);
}

// Dump a string vector as a JSON array.
// 将字符串向量写成 JSON 数组.
static void dump_string_vector (FILE *file,
                                const std::vector<std::string> &strings) {
  fputc ('[', file);
  for (size_t i = 0; i < strings.size (); i++) {
    if (i)
      fputc (',', file);
    fprintf (file, "\"%s\"", strings[i].c_str ());
  }
  fputc (']', file);
}

// Encode compact clause metadata flags for the dump stream.
// 为 dump 流编码紧凑的子句元数据标志.
static int clause_flags (Internal *internal, Clause *c) {
  int res = 0;
  if (c->redundant)
    res |= 1;
  if (c->hyper)
    res |= 2;
  if (internal->likely_to_be_kept_clause (c))
    res |= 4;
  return res;
}

} // namespace

// Enable writing a standalone lookahead dump to the given path.
// 启用将一次独立 lookahead dump 写入指定路径.
void Internal::enable_lookahead_dump (const char *path) {
  assert (path);
  dump_lookahead = true;
  dump_lookahead_path = path;
}

// Select the JSONL schema profile used by lookahead dump.
// 选择 lookahead dump 使用的 JSONL schema profile.
void Internal::set_lookahead_dump_profile (const char *profile) {
  assert (profile);
  if (!strcmp (profile, "full"))
    dump_lookahead_profile = LOOKAHEAD_DUMP_FULL;
  else if (!strcmp (profile, "compact"))
    dump_lookahead_profile = LOOKAHEAD_DUMP_COMPACT;
  else if (!strcmp (profile, "lite"))
    dump_lookahead_profile = LOOKAHEAD_DUMP_LITE;
  else if (!strcmp (profile, "stats"))
    dump_lookahead_profile = LOOKAHEAD_DUMP_STATS;
  else if (!strcmp (profile, "tabular"))
    dump_lookahead_profile = LOOKAHEAD_DUMP_TABULAR;
  else
    error ("invalid lookahead dump profile '%s'", profile);
}

// Select the lookahead dump generation mode.
// 选择 lookahead dump 生成模式.
void Internal::set_lookahead_dump_mode (const char *mode) {
  assert (mode);
  if (!strcmp (mode, "legacy"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_LEGACY;
  else if (!strcmp (mode, "v2-basic"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_V2_BASIC;
  else if (!strcmp (mode, "v2-cube"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_V2_CUBE;
  else if (!strcmp (mode, "v2-lookahead-eval"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_V2_EVAL;
  else if (!strcmp (mode, "v2-full"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_V2_FULL;
  else if (!strcmp (mode, "v2-eval-only"))
    dump_lookahead_mode = LOOKAHEAD_DUMP_V2_EVAL_ONLY;
  else
    error ("invalid lookahead dump mode '%s'", mode);
}

// Store bounded first-decision evaluation options for v2 dump modes.
// 保存 v2 dump mode 使用的有界第一次决策评估选项.
void Internal::set_lookahead_dump_eval_options (int k, double k_rate,
                                                int conflicts,
                                                int timeout_ms,
                                                int random) {
  dump_lookahead_eval_k = k;
  dump_lookahead_eval_k_rate = k_rate;
  dump_lookahead_eval_conflicts = conflicts;
  dump_lookahead_eval_timeout_ms = timeout_ms;
  dump_lookahead_eval_random = random;
}

// Store the original input description for fresh dump evaluation solvers.
// 保存 fresh dump evaluation solver 使用的原始输入描述.
void Internal::set_lookahead_dump_input (const char *path, int strict,
                                         bool simplify) {
  assert (path);
  dump_lookahead_input_path = path;
  dump_lookahead_input_strict = strict;
  dump_lookahead_input_simplify = simplify;
}

// Configure a dump-only forced first decision override.
// 配置仅用于 dump 的强制第一次决策覆盖.
void Internal::lookahead_dump_set_eval_override (int lit, int timeout_ms) {
  dump_eval_forced_first_lit = lit;
  dump_eval_first_decision_lit = 0;
  dump_eval_first_decision_trail_before = 0;
  dump_eval_first_decision_trail_gain = 0;
  dump_eval_first_decision_pending = false;
  dump_eval_forced_first_used = false;
  dump_eval_timed_out = false;
  dump_eval_timeout_deadline = timeout_ms > 0 ?
                                   absolute_real_time () + timeout_ms / 1000.0 :
                                   0;
}

// Clear the dump-only forced first decision override.
// 清除仅用于 dump 的强制第一次决策覆盖.
void Internal::lookahead_dump_clear_eval_override () {
  dump_eval_forced_first_lit = 0;
  dump_eval_timeout_deadline = 0;
}

// Run a bounded solve with an optional forced first decision.
// 使用可选强制第一次决策运行有界求解.
LookaheadDumpEvalResult Internal::lookahead_dump_evaluate_first_decision (
    int lit, int conflicts, int timeout_ms) {
  LookaheadDumpEvalResult res;
  memset (&res, 0, sizeof res);
  res.lit = lit;
  res.is_default = !lit;

  const double start = absolute_real_time ();
  lookahead_dump_set_eval_override (lit, timeout_ms);
  if (conflicts > 0)
    limit_conflicts (conflicts);
  const int status = solve (false);
  const double end = absolute_real_time ();
  res.status = status;
  res.conflicts = stats.conflicts;
  res.decisions = stats.decisions;
  res.propagations = stats.propagations.search;
  res.restarts = stats.restarts;
  res.time_ms = (int) ((end - start) * 1000.0 + 0.5);
  res.timeout = dump_eval_timed_out;
  res.decision_lit = dump_eval_first_decision_lit;
  res.first_decision_trail_gain = dump_eval_first_decision_trail_gain;
  res.first_conflict_depth = -1;
  res.first_learned_clause_size = -1;
  res.first_learned_lbd = -1;
  res.first_restart_conflicts = -1;
  res.first_restart_propagations = -1;
  lookahead_dump_clear_eval_override ();
  return res;
}

// Emit the compact schema record for legacy non-full dump profiles.
// 为 legacy 非 full dump profile 输出紧凑 schema 记录.
static void dump_schema (FILE *file, int profile) {
  if (profile == LOOKAHEAD_DUMP_FULL)
    return;
  if (profile == LOOKAHEAD_DUMP_COMPACT) {
    fputs ("{\"type\":\"schema\",\"profile\":\"compact\"," \
           "\"version\":1,\"events\":{" \
           "\"b\":[\"max_var\",\"original_vars\",\"current_vars\"," \
           "\"original_clauses\",\"current_clauses\"," \
           "\"irredundant_clauses\",\"redundant_clauses\",\"level\"," \
           "\"num_assigned\",\"trail_size\",\"propagated\"," \
           "\"probing_round\",\"conflicts\",\"assumptions\"]," \
           "\"k\":[\"result\",\"status\"],\"v\":[\"active\"]," \
           "\"c\":[\"id\",\"flags\",\"glue\",\"used\",\"lits\"]," \
           "\"a\":[\"trail\"]," \
           "\"p\":[\"lit\",\"ok\",\"trail_before\",\"trail_after\"," \
           "\"hbrs\",\"implied\"]," \
           "\"z\":[\"lit\"],\"e\":[]}," \
           "\"clause_flags\":{\"redundant\":1,\"hyper\":2," \
           "\"keep\":4}," \
           "\"assignment\":[\"lit\",\"level\",\"trail\",\"reason\"]}\n",
           file);
  } else if (profile == LOOKAHEAD_DUMP_LITE) {
    fputs ("{\"type\":\"schema\",\"profile\":\"lite\"," \
           "\"version\":1,\"events\":{" \
           "\"b\":[\"max_var\",\"original_vars\",\"current_vars\"," \
           "\"original_clauses\",\"current_clauses\"," \
           "\"irredundant_clauses\",\"redundant_clauses\",\"level\"," \
           "\"num_assigned\",\"trail_size\",\"propagated\"," \
           "\"probing_round\",\"conflicts\",\"assumptions\"]," \
           "\"k\":[\"result\",\"status\"],\"v\":[\"active\"]," \
           "\"c\":[\"id\",\"redundant\",\"lits\"]," \
           "\"a\":[\"trail\"]," \
           "\"p\":[\"lit\",\"ok\",\"trail_before\",\"trail_after\"," \
           "\"hbrs\",\"implied_count\"]," \
           "\"z\":[\"lit\"],\"e\":[]}," \
           "\"assignment\":[\"lit\",\"level\",\"trail\",\"reason\"]}\n",
           file);
  } else {
    fputs ("{\"type\":\"schema\",\"profile\":\"stats\"," \
           "\"version\":1,\"events\":{" \
           "\"b\":[\"max_var\",\"original_vars\",\"current_vars\"," \
           "\"original_clauses\",\"current_clauses\"," \
           "\"irredundant_clauses\",\"redundant_clauses\",\"level\"," \
           "\"num_assigned\",\"trail_size\",\"propagated\"," \
           "\"probing_round\",\"conflicts\",\"assumptions\"]," \
           "\"k\":[\"result\",\"status\"]," \
           "\"p\":[\"lit\",\"ok\",\"trail_before\",\"trail_after\"," \
           "\"hbrs\",\"implied_count\"]," \
           "\"z\":[\"lit\"],\"e\":[]}}\n",
           file);
  }
}

// Emit the v2 schema record for full, compact, or tabular output.
// 为 full, compact, 或 tabular 输出 v2 schema 记录.
static void dump_v2_schema (FILE *file, int profile, int mode) {
  const char *profile_name = profile == LOOKAHEAD_DUMP_FULL ?
                                 "full" :
                             profile == LOOKAHEAD_DUMP_TABULAR ?
                                 "tabular" :
                                 "compact";
  const char *mode_name = mode == LOOKAHEAD_DUMP_V2_BASIC ?
                              "v2-basic" :
                          mode == LOOKAHEAD_DUMP_V2_CUBE ?
                              "v2-cube" :
                          mode == LOOKAHEAD_DUMP_V2_EVAL ?
                              "v2-lookahead-eval" :
                          mode == LOOKAHEAD_DUMP_V2_EVAL_ONLY ?
                              "v2-eval-only" :
                              "v2-full";
  if (profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (file,
             "{\"type\":\"schema\",\"version\":2,"
             "\"profile\":\"%s\",\"mode\":\"%s\","
             "\"extends\":\"lookahead-dump-v1\"}\n",
             profile_name, mode_name);
  } else if (profile == LOOKAHEAD_DUMP_COMPACT) {
    fprintf (file,
             "{\"type\":\"schema\",\"version\":2,"
             "\"profile\":\"%s\",\"mode\":\"%s\","
             "\"extends\":\"lookahead-dump-v1\",\"events\":{"
             "\"c\":[\"id\",\"flags\",\"glue\",\"used\","
             "\"satisfied\",\"lits\"],"
             "\"p\":[\"lit\",\"ok\",\"trail_before\","
             "\"trail_after\",\"hbrs\",\"implied\","
             "\"probe_index\"],"
             "\"le\":[\"probe_index\",\"lit\","
             "\"satisfied_clause_count\","
             "\"shortened_clause_count\","
             "\"new_binary_clause_count\","
             "\"new_unit_clause_count\"],"
             "\"d\":[\"lit\",\"is_default\",\"source\","
             "\"decision_lit\",\"status\",\"conflicts\","
             "\"decisions\",\"propagations\",\"restarts\","
             "\"time_ms\",\"timeout\","
             "\"first_decision_trail_gain\","
             "\"first_conflict_depth\","
             "\"first_learned_clause_size\","
             "\"first_learned_lbd\","
             "\"first_restart_conflicts\","
             "\"first_restart_propagations\"],"
             "\"m\":[\"probe_count\",\"effective_probe_count\","
             "\"decision_eval_count\"]}}\n",
             profile_name, mode_name);
  } else {
    fprintf (file,
             "{\"type\":\"schema\",\"version\":2,"
             "\"profile\":\"%s\",\"mode\":\"%s\","
             "\"extends\":\"lookahead-dump-v1\",\"events\":{"
             "\"begin\":[\"max_var\",\"original_vars\","
             "\"current_vars\",\"original_clauses\","
             "\"current_clauses\",\"irredundant_clauses\","
             "\"redundant_clauses\",\"level\",\"num_assigned\","
             "\"trail_size\",\"propagated\",\"probing_round\","
             "\"conflicts\",\"assumptions\"],"
             "\"check\":[\"result\",\"status\"],"
             "\"variables\":[\"active\"],"
             "\"clause\":[\"id\",\"flags\",\"glue\","
             "\"used\",\"satisfied\",\"lits\"],"
             "\"assignment\":[\"trail\"],"
             "\"probe\":[\"lit\",\"ok\",\"trail_before\","
             "\"trail_after\",\"hbrs\",\"implied\","
             "\"probe_index\"],"
             "\"literal_effects\":[\"probe_index\",\"lit\","
             "\"satisfied_clause_count\",\"shortened_clause_count\","
             "\"new_binary_clause_count\",\"new_unit_clause_count\"],"
             "\"decision_eval\":[\"lit\",\"is_default\","
             "\"source\",\"decision_lit\",\"status\","
             "\"conflicts\",\"decisions\",\"propagations\","
             "\"restarts\",\"time_ms\",\"timeout\","
             "\"first_decision_trail_gain\",\"first_conflict_depth\","
             "\"first_learned_clause_size\",\"first_learned_lbd\","
             "\"first_restart_conflicts\","
             "\"first_restart_propagations\"],"
             "\"summary\":[\"probe_count\","
             "\"effective_probe_count\",\"decision_eval_count\"],"
             "\"chosen\":[\"lit\"],\"end\":[]}}\n",
             profile_name, mode_name);
  }
}

// Return whether a clause is satisfied under the current assignment.
// 返回子句在当前赋值下是否已满足.
static bool clause_satisfied_now (Internal *internal, Clause *c) {
  for (const auto &lit : *c)
    if (internal->val (lit) > 0)
      return true;
  return false;
}

// Return whether a literal can be evaluated as a fresh decision candidate.
// 判断一个 literal 是否可以作为 fresh decision 候选被评估.
static bool eval_candidate_lit (Internal *internal, int lit) {
  return lit && internal->active (lit) && !internal->assumed (lit) &&
         !internal->assumed (-lit) && !internal->val (lit);
}

// Add a candidate source label, merging duplicate literals.
// 添加 candidate source 标签, 并合并重复 literal.
static bool add_eval_candidate (std::vector<DecisionEvalCandidate> &out,
                                int lit, const char *source) {
  if (!lit)
    return false;
  for (auto &candidate : out) {
    if (candidate.lit != lit)
      continue;
    std::string label (source);
    if (std::find (candidate.source.begin (), candidate.source.end (),
                   label) != candidate.source.end ())
      return false;
    candidate.source.push_back (label);
    return true;
  }
  DecisionEvalCandidate candidate;
  candidate.lit = lit;
  candidate.source.push_back (source);
  out.push_back (candidate);
  return true;
}

// Compute the non-default eval candidate limit after k and rate caps.
// 根据 k 和 rate 上限计算非默认 eval 候选数量限制.
static int eval_candidate_limit (Internal *internal) {
  int limit = internal->dump_lookahead_eval_k;
  if (internal->dump_lookahead_eval_k_rate > 0) {
    const int rate_limit =
        (int) ceil (2.0 * internal->max_var *
                    internal->dump_lookahead_eval_k_rate);
    if (limit > 0)
      limit = std::min (limit, rate_limit);
    else
      limit = rate_limit;
  }
  return limit;
}

// Append ranked static literal candidates from current live clauses.
// 从当前 live clauses 中追加静态排序的 literal 候选.
static void add_ranked_static_candidates (
    Internal *internal, std::vector<DecisionEvalCandidate> &out,
    const char *source, bool jw, bool moms, int limit) {
  std::vector<LiteralScore> scores;
  scores.reserve ((size_t) 2 * internal->max_var);
  for (int idx = 1; idx <= internal->max_var; idx++) {
    if (eval_candidate_lit (internal, idx)) {
      LiteralScore score;
      score.lit = idx;
      scores.push_back (score);
    }
    if (eval_candidate_lit (internal, -idx)) {
      LiteralScore score;
      score.lit = -idx;
      scores.push_back (score);
    }
  }

  int min_size = INT_MAX;
  if (moms) {
    for (Clause *c : internal->clauses) {
      if (c->garbage || clause_satisfied_now (internal, c))
        continue;
      int available = 0;
      for (const auto &lit : *c)
        if (eval_candidate_lit (internal, lit))
          available++;
      if (available > 0 && available < min_size)
        min_size = available;
    }
  }

  for (auto &score : scores) {
    for (Clause *c : internal->clauses) {
      if (c->garbage || clause_satisfied_now (internal, c))
        continue;
      bool contains = false;
      int available = 0;
      for (const auto &lit : *c) {
        if (eval_candidate_lit (internal, lit))
          available++;
        if (lit == score.lit)
          contains = true;
      }
      if (!contains)
        continue;
      if (moms) {
        if (available == min_size)
          score.score += 1;
      } else if (jw)
        score.score += pow (0.5, c->size);
      else
        score.score += 1;
    }
  }

  std::sort (scores.begin (), scores.end (),
             [] (const LiteralScore &a, const LiteralScore &b) {
               if (a.score != b.score)
                 return a.score > b.score;
               return a.lit < b.lit;
             });
  int added = 0;
  for (const auto &score : scores) {
    if (score.score <= 0)
      continue;
    if (limit > 0 && added >= limit)
      break;
    const bool tagged = add_eval_candidate (out, score.lit, source);
    if (tagged)
      added++;
  }
}

// Append deterministic random literal candidates as a control source.
// 追加确定性的随机 literal 候选作为 control 来源.
static void add_random_candidates (Internal *internal,
                                   std::vector<DecisionEvalCandidate> &out,
                                   int limit) {
  if (internal->dump_lookahead_eval_random <= 0)
    return;
  std::vector<int> lits;
  for (int idx = 1; idx <= internal->max_var; idx++) {
    if (eval_candidate_lit (internal, idx))
      lits.push_back (idx);
    if (eval_candidate_lit (internal, -idx))
      lits.push_back (-idx);
  }
  uint64_t state = 1469598103934665603ull ^ (uint64_t) internal->max_var ^
                   (uint64_t) internal->stats.conflicts;
  for (size_t i = lits.size (); i > 1; i--) {
    state = state * 1099511628211ull + 1469598103934665603ull;
    const size_t j = (size_t) (state % i);
    std::swap (lits[i - 1], lits[j]);
  }
  int added = 0;
  const int random_limit = limit > 0 ?
                               std::min (limit,
                                         internal->dump_lookahead_eval_random) :
                               internal->dump_lookahead_eval_random;
  for (int lit : lits) {
    if (random_limit > 0 && added >= random_limit)
      break;
    const bool tagged = add_eval_candidate (out, lit, "random");
    if (tagged)
      added++;
  }
}

// Count the remaining unassigned literals before and after a probe trail slice.
// 统计 probe trail slice 前后的剩余未赋值 literal 数量.
static void clause_probe_state (Internal *internal, Clause *c,
                                const std::vector<signed char> &delta,
                                bool &before_sat, bool &after_sat,
                                int &before_remaining,
                                int &after_remaining) {
  before_sat = after_sat = false;
  before_remaining = after_remaining = 0;
  for (const auto &lit : *c) {
    const int idx = abs (lit);
    const signed char assigned = idx < (int) delta.size () ? delta[idx] : 0;
    signed char before = assigned ? 0 : internal->val (lit);
    signed char after = assigned ? (assigned == (lit > 0 ? 1 : -1) ? 1 : -1) :
                                   internal->val (lit);
    if (before > 0)
      before_sat = true;
    if (after > 0)
      after_sat = true;
    if (!before)
      before_remaining++;
    if (!after)
      after_remaining++;
  }
}

// Compute post-propagation effects for one probe trail slice.
// 计算一次 probe trail slice 的传播后效果.
static LiteralEffects compute_literal_effects (Internal *internal,
                                               size_t before,
                                               size_t after) {
  LiteralEffects res;
  std::vector<signed char> delta ((size_t) internal->max_var + 1, 0);
  for (size_t i = before; i < after; i++) {
    const int lit = internal->trail[i];
    delta[abs (lit)] = lit > 0 ? 1 : -1;
  }
  for (Clause *c : internal->clauses) {
    if (c->garbage)
      continue;
    bool before_sat, after_sat;
    int before_remaining, after_remaining;
    clause_probe_state (internal, c, delta, before_sat, after_sat,
                        before_remaining, after_remaining);
    if (before_sat)
      continue;
    if (after_sat) {
      res.satisfied_clause_count++;
      continue;
    }
    if (after_remaining < before_remaining) {
      res.shortened_clause_count++;
      if (before_remaining > 2 && after_remaining == 2)
        res.new_binary_clause_count++;
      if (before_remaining > 1 && after_remaining == 1)
        res.new_unit_clause_count++;
    }
  }
  return res;
}

// Emit the begin record and the initial solver snapshot.
// 输出 begin 记录和初始求解器快照.
void Internal::dump_lookahead_begin () {
  assert (dump_lookahead);
  if (!dump_lookahead_file) {
    dump_lookahead_file = fopen (dump_lookahead_path.c_str (), "w");
    if (!dump_lookahead_file)
      error ("can not open lookahead dump file '%s'",
             dump_lookahead_path.c_str ());
  }

  FILE *file = dump_lookahead_file;
  const int profile = dump_lookahead_profile;
  const bool v2 = is_v2_mode (dump_lookahead_mode);

  if (v2)
    dump_v2_schema (file, profile, dump_lookahead_mode);
  else
    dump_schema (file, profile);

  int64_t current_vars = 0;
  for (int idx = 1; idx <= max_var; idx++)
    if (active (idx))
      current_vars++;

  int64_t current_clauses = 0;
  int64_t irredundant_clauses = 0;
  int64_t redundant_clauses = 0;
  for (Clause *c : clauses) {
    if (c->garbage)
      continue;
    current_clauses++;
    if (c->redundant)
      redundant_clauses++;
    else
      irredundant_clauses++;
  }

  const int64_t original_vars = dump_lookahead_original_vars >= 0 ?
                                    dump_lookahead_original_vars :
                                    stats.variables_original;
  const int64_t original_clauses = dump_lookahead_original_clauses >= 0 ?
                                       dump_lookahead_original_clauses :
                                       stats.added.irredundant;

  if (profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (file,
             "{\"type\":\"begin\",\"max_var\":%d,"
             "\"original_vars\":%" PRId64 ","
             "\"current_vars\":%" PRId64 ","
             "\"original_clauses\":%" PRId64 ","
             "\"current_clauses\":%" PRId64 ","
             "\"irredundant_clauses\":%" PRId64 ","
             "\"redundant_clauses\":%" PRId64 ","
             "\"level\":%d,\"num_assigned\":%zu,"
             "\"trail_size\":%zu,\"propagated\":%zu,"
             "\"probing_round\":%" PRId64 ","
             "\"conflicts\":%" PRId64 ",\"assumptions\":",
             max_var, original_vars, current_vars, original_clauses,
             current_clauses, irredundant_clauses, redundant_clauses,
             level, num_assigned, trail.size (), propagated,
             stats.probingrounds, stats.conflicts);
    dump_int_vector (file, assumptions);
    fputs ("}\n", file);

    fputs ("{\"type\":\"check\",\"result\":0,"
           "\"status\":\"unknown\"}\n",
           file);

    fputs ("{\"type\":\"variables\",\"active\":[", file);
    bool first = true;
    for (int idx = 1; idx <= max_var; idx++) {
      if (!active (idx))
        continue;
      if (!first)
        fputc (',', file);
      first = false;
      fprintf (file, "%d", idx);
    }
    fputs ("]}\n", file);

    for (Clause *c : clauses) {
      if (c->garbage)
        continue;
      fprintf (file,
               "{\"type\":\"clause\",\"id\":%" PRId64 ","
               "\"redundant\":%s,\"glue\":%d,\"used\":%u,"
               "\"hyper\":%s,\"keep\":%s,",
               c->id, c->redundant ? "true" : "false", c->glue,
               c->used, c->hyper ? "true" : "false",
               likely_to_be_kept_clause (c) ? "true" : "false");
      if (v2)
        fprintf (file, "\"satisfied\":%s,",
                 clause_satisfied_now (this, c) ? "true" : "false");
      fprintf (file, "\"size\":%d,\"lits\":", c->size);
      dump_lits (file, c);
      fputs ("}\n", file);
    }

    fputs ("{\"type\":\"assignment\",\"trail\":[", file);
    for (size_t i = 0; i < trail.size (); i++) {
      int lit = trail[i];
      const Var &v = var (lit);
      uint64_t reason_id = 0;
      if (v.reason && v.reason != external_reason)
        reason_id = v.reason->id;
      if (i)
        fputc (',', file);
      fprintf (file,
               "{\"lit\":%d,\"level\":%d,\"trail\":%d,"
               "\"reason\":%" PRIu64 "}",
               lit, v.level, v.trail, reason_id);
    }
    fputs ("]}\n", file);
  } else {
    fprintf (file,
             "[\"%s\",%d,%" PRId64 ",%" PRId64 ",%" PRId64 ","
             "%" PRId64 ",%" PRId64 ",%" PRId64 ",%d,%zu,%zu,%zu,"
             "%" PRId64 ",%" PRId64 ",",
             row_label (profile, "b", "begin"), max_var, original_vars,
             current_vars, original_clauses,
             current_clauses, irredundant_clauses, redundant_clauses,
             level, num_assigned, trail.size (), propagated,
             stats.probingrounds, stats.conflicts);
    dump_int_vector (file, assumptions);
    fputs ("]\n", file);

    fprintf (file, "[\"%s\",0,\"unknown\"]\n",
             row_label (profile, "k", "check"));

    if (profile == LOOKAHEAD_DUMP_COMPACT ||
        profile == LOOKAHEAD_DUMP_TABULAR ||
        profile == LOOKAHEAD_DUMP_LITE) {
      fprintf (file, "[\"%s\",[",
               row_label (profile, "v", "variables"));
      bool first = true;
      for (int idx = 1; idx <= max_var; idx++) {
        if (!active (idx))
          continue;
        if (!first)
          fputc (',', file);
        first = false;
        fprintf (file, "%d", idx);
      }
      fputs ("]]\n", file);

      for (Clause *c : clauses) {
        if (c->garbage)
          continue;
        if (profile == LOOKAHEAD_DUMP_COMPACT ||
            profile == LOOKAHEAD_DUMP_TABULAR) {
          fprintf (file, "[\"%s\",%" PRId64 ",%d,%d,%u,",
                   row_label (profile, "c", "clause"), c->id,
                   clause_flags (this, c), c->glue, c->used);
          if (v2)
            fprintf (file, "%d,", clause_satisfied_now (this, c) ? 1 : 0);
        } else {
          fprintf (file, "[\"c\",%" PRId64 ",%d,", c->id,
                   c->redundant ? 1 : 0);
        }
        dump_lits (file, c);
        fputs ("]\n", file);
      }

      fprintf (file, "[\"%s\",[",
               row_label (profile, "a", "assignment"));
      for (size_t i = 0; i < trail.size (); i++) {
        int lit = trail[i];
        const Var &v = var (lit);
        uint64_t reason_id = 0;
        if (v.reason && v.reason != external_reason)
          reason_id = v.reason->id;
        if (i)
          fputc (',', file);
        fprintf (file, "[%d,%d,%d,%" PRIu64 "]", lit, v.level,
                 v.trail, reason_id);
      }
      fputs ("]]\n", file);
    }
  }
}

// Emit one probe result record, including the temporary trail slice.
// 输出一条 probe 结果记录, 包括临时 trail 切片.
void Internal::dump_lookahead_probe_result (int probe, bool ok,
                                            size_t before, size_t after,
                                            int hbrs) {
  assert (dump_lookahead);
  assert (dump_lookahead_file);
  FILE *file = dump_lookahead_file;
  const int profile = dump_lookahead_profile;
  const bool v2 = is_v2_mode (dump_lookahead_mode);
  if (profile == LOOKAHEAD_DUMP_FULL) {
    if (v2)
      fprintf (file,
               "{\"type\":\"probe\",\"probe_index\":%d,"
               "\"lit\":%d,\"ok\":%s,\"failed\":%s,"
               "\"trail_before\":%zu,\"trail_after\":%zu,"
               "\"hbrs\":%d,\"implied_count\":%zu,"
               "\"implied\":[",
               dump_lookahead_probe_count, probe, ok ? "true" : "false",
               ok ? "false" : "true", before, after, hbrs, after - before);
    else
      fprintf (file,
               "{\"type\":\"probe\",\"lit\":%d,\"ok\":%s,"
               "\"failed\":%s,\"trail_before\":%zu,"
               "\"trail_after\":%zu,\"hbrs\":%d,"
               "\"implied_count\":%zu,\"implied\":[",
               probe, ok ? "true" : "false", ok ? "false" : "true",
               before, after, hbrs, after - before);
    for (size_t i = before; i < after; i++) {
      if (i != before)
        fputc (',', file);
      fprintf (file, "%d", trail[i]);
    }
    fputs ("]}\n", file);
  } else if (profile == LOOKAHEAD_DUMP_COMPACT ||
             profile == LOOKAHEAD_DUMP_TABULAR) {
    fprintf (file, "[\"%s\",%d,%d,%zu,%zu,%d,[",
             row_label (profile, "p", "probe"), probe, ok ? 1 : 0,
             before, after, hbrs);
    for (size_t i = before; i < after; i++) {
      if (i != before)
        fputc (',', file);
      fprintf (file, "%d", trail[i]);
    }
    if (v2)
      fprintf (file, "],%d]\n", dump_lookahead_probe_count);
    else
      fputs ("]]\n", file);
  } else {
    fprintf (file, "[\"p\",%d,%d,%zu,%zu,%d,%zu]\n", probe,
             ok ? 1 : 0, before, after, hbrs, after - before);
  }
}

// Emit literal effect counters for the most recent v2 probe.
// 输出最近一次 v2 probe 的 literal effect 计数.
static void dump_literal_effects (Internal *internal, int probe,
                                  int probe_index,
                                  const LiteralEffects &effects) {
  FILE *file = internal->dump_lookahead_file;
  if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (file,
             "{\"type\":\"literal_effects\","
             "\"probe_index\":%d,\"lit\":%d,"
             "\"satisfied_clause_count\":%d,"
             "\"shortened_clause_count\":%d,"
             "\"new_binary_clause_count\":%d,"
             "\"new_unit_clause_count\":%d}\n",
             probe_index, probe, effects.satisfied_clause_count,
             effects.shortened_clause_count, effects.new_binary_clause_count,
             effects.new_unit_clause_count);
  } else if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_COMPACT ||
             internal->dump_lookahead_profile == LOOKAHEAD_DUMP_TABULAR) {
    fprintf (file, "[\"%s\",%d,%d,%d,%d,%d,%d]\n",
             row_label (internal->dump_lookahead_profile, "le",
                        "literal_effects"),
             probe_index, probe,
             effects.satisfied_clause_count,
             effects.shortened_clause_count,
             effects.new_binary_clause_count,
             effects.new_unit_clause_count);
  }
}

// Emit one first-decision evaluation record.
// 输出一条第一次决策评估记录.
static void dump_decision_eval (Internal *internal,
                                const LookaheadDumpEvalResult &res,
                                const std::vector<std::string> &source) {
  FILE *file = internal->dump_lookahead_file;
  if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (file,
             "{\"type\":\"decision_eval\",\"lit\":%d,"
             "\"is_default\":%s,\"source\":",
             res.lit, res.is_default ? "true" : "false");
    dump_string_vector (file, source);
    fprintf (file,
             ",\"decision_lit\":%d,\"status\":%d,"
             "\"conflicts\":%" PRId64 ","
             "\"decisions\":%" PRId64 ","
             "\"propagations\":%" PRId64 ","
             "\"restarts\":%" PRId64 ",\"time_ms\":%d,"
             "\"timeout\":%s,"
             "\"first_decision_trail_gain\":%d,"
             "\"first_conflict_depth\":%d,"
             "\"first_learned_clause_size\":%d,"
             "\"first_learned_lbd\":%d,"
             "\"first_restart_conflicts\":%d,"
             "\"first_restart_propagations\":%" PRId64 "}\n",
             res.decision_lit, res.status, res.conflicts, res.decisions,
             res.propagations, res.restarts, res.time_ms,
             res.timeout ? "true" : "false",
             res.first_decision_trail_gain, res.first_conflict_depth,
             res.first_learned_clause_size, res.first_learned_lbd,
             res.first_restart_conflicts, res.first_restart_propagations);
  } else if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_COMPACT ||
             internal->dump_lookahead_profile == LOOKAHEAD_DUMP_TABULAR) {
    fprintf (file, "[\"%s\",%d,%d,",
             row_label (internal->dump_lookahead_profile, "d",
                        "decision_eval"),
             res.lit, res.is_default ? 1 : 0);
    dump_string_vector (file, source);
    fprintf (file,
             ",%d,%d,%" PRId64 ",%" PRId64 ",%" PRId64 ","
             "%" PRId64 ",%d,%d,%d,%d,%d,%d,%d,%" PRId64 "]\n",
             res.decision_lit, res.status, res.conflicts, res.decisions,
             res.propagations, res.restarts, res.time_ms,
             res.timeout ? 1 : 0, res.first_decision_trail_gain,
             res.first_conflict_depth, res.first_learned_clause_size,
             res.first_learned_lbd, res.first_restart_conflicts,
             res.first_restart_propagations);
  }
}

// Emit the v2 summary record before the final end marker.
// 在最终 end 标记前输出 v2 summary 记录.
static void dump_v2_summary (Internal *internal) {
  FILE *file = internal->dump_lookahead_file;
  if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (file,
             "{\"type\":\"summary\",\"probe_count\":%d,"
             "\"effective_probe_count\":%d,"
             "\"decision_eval_count\":%d}\n",
             internal->dump_lookahead_probe_count,
             internal->dump_lookahead_effective_probe_count,
             internal->dump_lookahead_decision_eval_count);
  } else if (internal->dump_lookahead_profile == LOOKAHEAD_DUMP_COMPACT ||
             internal->dump_lookahead_profile == LOOKAHEAD_DUMP_TABULAR) {
    fprintf (file, "[\"%s\",%d,%d,%d]\n",
             row_label (internal->dump_lookahead_profile, "m", "summary"),
             internal->dump_lookahead_probe_count,
             internal->dump_lookahead_effective_probe_count,
             internal->dump_lookahead_decision_eval_count);
  }
}

// Emit the chosen literal and close the lookahead dump stream.
// 输出选中的 literal, 并关闭 lookahead dump 流.
void Internal::dump_lookahead_end (int chosen) {
  assert (dump_lookahead);
  assert (dump_lookahead_file);
  if (chosen == INT_MIN)
    chosen = 0;
  if (is_v2_mode (dump_lookahead_mode))
    dump_v2_summary (this);
  if (dump_lookahead_profile == LOOKAHEAD_DUMP_FULL) {
    fprintf (dump_lookahead_file, "{\"type\":\"chosen\",\"lit\":%d}\n",
             chosen);
    fputs ("{\"type\":\"end\"}\n", dump_lookahead_file);
  } else {
    fprintf (dump_lookahead_file, "[\"%s\",%d]\n",
             row_label (dump_lookahead_profile, "z", "chosen"), chosen);
    fprintf (dump_lookahead_file, "[\"%s\"]\n",
             row_label (dump_lookahead_profile, "e", "end"));
  }
  if (fclose (dump_lookahead_file))
    error ("can not close lookahead dump file '%s'",
           dump_lookahead_path.c_str ());
  dump_lookahead_file = 0;
}

// Add the current residual clause database to a fresh evaluation solver.
// 将当前 residual clause database 加入 fresh evaluation solver.
static void add_eval_snapshot_to_solver (Internal *internal, Solver &solver) {
  solver.resize (internal->max_var);
  for (Clause *c : internal->clauses) {
    if (c->garbage)
      continue;
    std::vector<int> residual;
    bool satisfied = false;
    for (const auto lit : *c) {
      const signed char value = internal->val (lit);
      if (value > 0) {
        satisfied = true;
        break;
      }
      if (!value)
        residual.push_back (lit);
    }
    if (satisfied)
      continue;
    for (const int residual_lit : residual)
      solver.add (residual_lit);
    solver.add (0);
  }
}

// Run one fresh solver evaluation and emit it to the current dump stream.
// 运行一次 fresh solver evaluation 并写入当前 dump 流.
static bool run_eval_candidate (Internal *internal, int lit,
                                const std::vector<std::string> &source) {
  Solver solver;
  (void) solver.set ("lucky", 0);
  (void) solver.set ("walk", 0);
  add_eval_snapshot_to_solver (internal, solver);
  LookaheadDumpEvalResult res = solver.lookahead_dump_evaluate_first_decision (
      lit, internal->dump_lookahead_eval_conflicts,
      internal->dump_lookahead_eval_timeout_ms);
  if (lit && res.decision_lit != lit)
    return false;
  dump_decision_eval (internal, res, source);
  internal->dump_lookahead_decision_eval_count++;
  return true;
}

// Run bounded first-decision evaluation for selected lookahead candidates.
// 对选中的 lookahead 候选运行有界第一次决策评估.
static void run_decision_evaluations (Internal *internal,
                                      const std::vector<ProbeRecord> &records,
                                      int chosen) {
  if (!mode_has_eval (internal->dump_lookahead_mode))
    return;

  if (run_eval_candidate (internal, 0, {"default"})) {
    // The default run is intentionally not counted against eval_k.
  }

  const int limit = eval_candidate_limit (internal);
  std::vector<DecisionEvalCandidate> candidates;

  if (eval_candidate_lit (internal, chosen))
    add_eval_candidate (candidates, chosen, "lookahead_chosen");

  std::vector<ProbeRecord> sorted = records;
  std::sort (sorted.begin (), sorted.end (),
             [] (const ProbeRecord &a, const ProbeRecord &b) {
               if (a.extra_implied_count != b.extra_implied_count)
                 return a.extra_implied_count > b.extra_implied_count;
               if (a.effects.new_unit_clause_count !=
                   b.effects.new_unit_clause_count)
                 return a.effects.new_unit_clause_count >
                        b.effects.new_unit_clause_count;
               if (a.effects.new_binary_clause_count !=
                   b.effects.new_binary_clause_count)
                 return a.effects.new_binary_clause_count >
                        b.effects.new_binary_clause_count;
               return a.effects.satisfied_clause_count >
                      b.effects.satisfied_clause_count;
             });

  int lookahead_topk_added = 0;
  for (const auto &record : sorted) {
    if (limit > 0 && lookahead_topk_added >= limit)
      break;
    if (eval_candidate_lit (internal, record.lit)) {
      const bool tagged =
          add_eval_candidate (candidates, record.lit, "lookahead_topk");
      if (tagged)
        lookahead_topk_added++;
    }
  }

  add_ranked_static_candidates (internal, candidates, "occurrence_topk",
                                false, false, limit);
  add_ranked_static_candidates (internal, candidates, "jw_topk", true,
                                false, limit);
  add_ranked_static_candidates (internal, candidates, "moms_topk", false,
                                true, limit);
  add_random_candidates (internal, candidates, limit);

  for (const auto &candidate : candidates)
    run_eval_candidate (internal, candidate.lit, candidate.source);
}

// Run lookahead probing with JSONL dump events enabled.
// 运行带 JSONL dump 事件输出的 lookahead probing.
int Internal::lookahead_dump_run_basic (bool eval) {

  dump_lookahead_probe_count = 0;
  dump_lookahead_effective_probe_count = 0;
  dump_lookahead_decision_eval_count = 0;

  if (!active ()) {
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return 0;
  }

  MSG ("lookahead-probe-round %" PRId64
       " without propagations limit and %zu assumptions",
       stats.probingrounds, assumptions.size ());

  termination_forced = false;

#ifndef QUIET
  int old_failed = stats.failed;
  int64_t old_probed = stats.probed;
#endif
  int64_t old_hbrs = stats.hbrs;

  if (unsat) {
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return INT_MIN;
  }
  if (level)
    backtrack ();
  if (!propagate ()) {
    MSG ("empty clause before probing");
    learn_empty_clause ();
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return INT_MIN;
  }

  if (terminating_asked ()) {
    int res = most_occurring_literal ();
    dump_lookahead_begin ();
    dump_lookahead_end (res);
    return res;
  }

  decompose ();

  if (ternary ())
    decompose ();

  mark_duplicated_binary_clauses_as_garbage ();

  lim.conflicts = -1;

  if (!probes.empty ())
    lookahead_flush_probes ();

  for (int idx = 1; idx <= max_var; idx++)
    propfixed (idx) = propfixed (-idx) = -1;

  assert (unsat || propagated == trail.size ());
  propagated = propagated2 = trail.size ();

  int probe;
  int res = most_occurring_literal ();
  int max_hbrs = -1;
  std::vector<ProbeRecord> records;

  set_mode (PROBE);

  MSG ("unsat = %d, terminating_asked () = %d ", unsat,
       terminating_asked ());
  init_probehbr_lrat ();
  dump_lookahead_begin ();
  while (!unsat && !terminating_asked () &&
         (probe = lookahead_next_probe ())) {
    stats.probed++;
    int hbrs;
    size_t before = trail.size ();

    probe_assign_decision (probe);
    bool ok = probe_propagate ();
    hbrs = ok ? trail.size () : 0;
    const size_t after = trail.size ();
    const int probe_index = dump_lookahead_probe_count;
    LiteralEffects effects;
    if (is_v2_mode (dump_lookahead_mode))
      effects = compute_literal_effects (this, before, after);
    dump_lookahead_probe_result (probe, ok, before, after, hbrs);
    if (is_v2_mode (dump_lookahead_mode))
      dump_literal_effects (this, probe, probe_index, effects);

    ProbeRecord record;
    record.lit = probe;
    record.probe_index = probe_index;
    record.trail_before = before;
    record.trail_after = after;
    record.hbrs = hbrs;
    record.extra_implied_count = after > before ? (int) (after - before - 1) : 0;
    record.effects = effects;
    records.push_back (record);
    dump_lookahead_probe_count++;
    if (record.extra_implied_count > 0)
      dump_lookahead_effective_probe_count++;

    if (ok)
      backtrack ();
    else
      failed_literal (probe);
    clean_probehbr_lrat ();
    if (max_hbrs < hbrs ||
        (max_hbrs == hbrs &&
         internal->bumped (probe) > internal->bumped (res))) {
      res = probe;
      max_hbrs = hbrs;
    }
  }

  reset_mode (PROBE);

  if (unsat) {
    MSG ("probing derived empty clause");
    res = INT_MIN;
  } else if (propagated < trail.size ()) {
    MSG ("probing produced %zd units",
         (size_t) (trail.size () - propagated));
    if (!propagate ()) {
      MSG ("propagating units after probing results in empty clause");
      learn_empty_clause ();
      res = INT_MIN;
    } else
      sort_watches ();
  }

#ifndef QUIET
  int failed = stats.failed - old_failed;
  int64_t probed = stats.probed - old_probed;
#endif
  int64_t hbrs = stats.hbrs - old_hbrs;

  MSG ("lookahead-probe-round %" PRId64 " probed %" PRId64
       " and found %d failed literals",
       stats.probingrounds, probed, failed);

  if (hbrs)
    PHASE ("lookahead-probe-round", stats.probingrounds,
           "found %" PRId64 " hyper binary resolvents", hbrs);

  MSG ("lookahead literal %d with %d\n", res, max_hbrs);

  if (eval)
    run_decision_evaluations (this, records, res == INT_MIN ? 0 : res);

  dump_lookahead_end (res);

  return res;
}

// Run only bounded first-decision evaluation without lookahead probes.
// 只运行有界第一次决策评估, 不执行 lookahead probe.
int Internal::lookahead_dump_run_eval_only () {

  dump_lookahead_probe_count = 0;
  dump_lookahead_effective_probe_count = 0;
  dump_lookahead_decision_eval_count = 0;

  if (!active ()) {
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return 0;
  }

  termination_forced = false;

  if (unsat) {
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return INT_MIN;
  }
  if (level)
    backtrack ();
  if (!propagate ()) {
    MSG ("empty clause before eval-only dump");
    learn_empty_clause ();
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return INT_MIN;
  }

  if (terminating_asked ()) {
    dump_lookahead_begin ();
    dump_lookahead_end (0);
    return 0;
  }

  decompose ();

  if (ternary ())
    decompose ();

  mark_duplicated_binary_clauses_as_garbage ();

  lim.conflicts = -1;
  assert (unsat || propagated == trail.size ());
  propagated = propagated2 = trail.size ();

  dump_lookahead_begin ();
  std::vector<ProbeRecord> no_records;
  run_decision_evaluations (this, no_records, 0);
  dump_lookahead_end (0);

  return 0;
}

// Run one lookahead dump according to the configured mode.
// 根据配置的 mode 运行一次 lookahead dump.
int Internal::lookahead_dump_run () {
  if (mode_eval_only (dump_lookahead_mode))
    return lookahead_dump_run_eval_only ();
  if (mode_has_cube (dump_lookahead_mode))
    return lookahead_dump_run_cube (mode_has_eval (dump_lookahead_mode));
  return lookahead_dump_run_basic (mode_has_eval (dump_lookahead_mode));
}

// Run cube fallback if the parent dump has no effective probe.
// 当父 dump 没有有效 probe 时运行 cube fallback.
int Internal::lookahead_dump_run_cube (bool eval) {
  const std::string parent_path = dump_lookahead_path;
  const int parent_mode = dump_lookahead_mode;
  dump_lookahead_mode = eval ? LOOKAHEAD_DUMP_V2_EVAL : LOOKAHEAD_DUMP_V2_BASIC;
  int res = lookahead_dump_run_basic (eval);
  const int chosen = res == INT_MIN ? 0 : res;
  const bool effective = dump_lookahead_effective_probe_count > 0;
  dump_lookahead_mode = parent_mode;
  if (effective || !chosen)
    return res;

  (void) remove (parent_path.c_str ());
  if (dump_lookahead_input_path.empty ())
    return res;

  const int children[2] = {chosen, -chosen};
  for (int child_lit : children) {
    Solver solver;
    int vars = 0;
    bool incremental = false;
    std::vector<int> cubes;
    const char *err = solver.read_dimacs (dump_lookahead_input_path.c_str (),
                                          vars, dump_lookahead_input_strict,
                                          incremental, cubes);
    if (err)
      continue;
    solver.assume (child_lit);
    const std::string child_path = cube_child_path (parent_path, child_lit);
    solver.enable_lookahead_dump (child_path.c_str ());
    solver.set_lookahead_dump_profile (
        dump_lookahead_profile == LOOKAHEAD_DUMP_COMPACT ?
            "compact" :
        dump_lookahead_profile == LOOKAHEAD_DUMP_TABULAR ?
            "tabular" :
            "full");
    solver.set_lookahead_dump_mode ("v2-basic");
    solver.set_lookahead_dump_input (dump_lookahead_input_path.c_str (),
                                     dump_lookahead_input_strict,
                                     dump_lookahead_input_simplify);
    if (dump_lookahead_input_simplify)
      (void) solver.simplify ();
    (void) solver.lookahead_for_dump ();
  }
  return res;
}

// Preserve the previous dump probing entry point for legacy callers.
// 为 legacy 调用者保留旧的 dump probing 入口.
int Internal::lookahead_probing_for_dump () { return lookahead_dump_run (); }

} // namespace CaDiCaL
