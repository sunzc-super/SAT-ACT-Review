#include "neuro_client.hpp"

#include <cmath>
#include <cstdio>
#include <grpc/grpc.h>
#include <grpc/impl/channel_arg_names.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

namespace {

satact_trace::ResponsePayload response_payload_code (
    const NeuroSATConfig &cfg) {
    return cfg.response_payload == "full"
               ? satact_trace::RESPONSE_PAYLOAD_FULL
               : satact_trace::RESPONSE_PAYLOAD_COMPACT;
}

bool fill_request (
    satact_trace::BranchRequest &args, const NeuroSATConfig &cfg,
    const CaDiCaL::Solver::SATActSnapshot &snapshot) {
    const int n_vars = snapshot.n_vars;
    if (n_vars <= 0 || snapshot.clauses.empty () ||
        snapshot.assignment_value.size () != (size_t) n_vars ||
        snapshot.assignment_level.size () != (size_t) n_vars ||
        snapshot.candidate_variable.size () != (size_t) n_vars)
        return false;

    args.set_n_vars (n_vars);
    args.set_n_clauses ((int) snapshot.clauses.size ());
    args.set_decision_level (snapshot.decision_level);
    args.set_model_variant (cfg.model_variant);
    args.set_response_payload (response_payload_code (cfg));

    int candidates = 0;
    for (int idx = 0; idx < n_vars; ++idx) {
        const int value = (int) snapshot.assignment_value[idx];
        if (value < -1 || value > 1)
            return false;
        const bool candidate = snapshot.candidate_variable[idx];
        if (candidate && value)
            return false;
        args.add_assignment_value (value);
        args.add_assignment_level (snapshot.assignment_level[idx]);
        args.add_candidate_variable (candidate);
        candidates += candidate ? 1 : 0;
    }
    if (!candidates)
        return false;

    for (size_t clause = 0; clause < snapshot.clauses.size (); ++clause) {
        if (snapshot.clauses[clause].empty ())
            return false;
        for (const int lit : snapshot.clauses[clause]) {
            const int var = lit < 0 ? -lit : lit;
            if (var < 1 || var > n_vars)
                return false;
            args.add_c_idxs ((int) clause);
            args.add_l_idxs (lit > 0 ? 2 * (var - 1)
                                     : 2 * (var - 1) + 1);
        }
    }
    return args.c_idxs_size () > 0;
}

void copy_decision (const satact_trace::BranchDecision &source,
                    NeuroDecisionData &target) {
    target.success = source.success ();
    target.n_secs_inference = source.n_secs_inference ();
    target.model_variant = source.model_variant ();
    target.action = (int) source.action ();
    target.selected_literal_index = source.selected_literal_index ();
    target.selected_log_probability = source.selected_log_probability ();
    target.response_payload = (int) source.response_payload ();
    target.action_logits_size = source.action_logits_size ();
    target.action_logits_finite = true;
    for (const float value : source.action_logits ())
        if (!std::isfinite (value)) {
            target.action_logits_finite = false;
            break;
        }
}

} // namespace

NeuroSATClient::NeuroSATClient (const NeuroSATConfig &config) : cfg (config) {
    if (cfg.n_secs_pause < 0.0)
        cfg.n_secs_pause = 0.0;
    if (cfg.n_secs_pause_inc <= 0.0)
        cfg.n_secs_pause_inc = 1.0;
}

bool NeuroSATClient::enabled () const {
    return cfg.mode == "SATACT";
}

bool NeuroSATClient::ready_to_call () {
    if (!enabled () || !stub || cfg.decide_strategy != "FIRST")
        return false;
    if (rpc_attempts >= cfg.neuro_calls) {
        budget_skips++;
        return false;
    }
    if (std::chrono::steady_clock::now () < t_next) {
        pause_skips++;
        return false;
    }
    return true;
}

bool NeuroSATClient::variant_allows_defer () const {
    return false;
}

void NeuroSATClient::begin_solve () {
    decision_opportunities = 0;
    rpc_attempts = 0;
    rpc_successes = 0;
    rpc_errors = 0;
    invalid_responses = 0;
    literal_actions = 0;
    defer_actions = 0;
    preflight_skips = 0;
    pause_skips = 0;
    budget_skips = 0;
    n_secs_inference = 0.0;
    n_secs_wait = 0.0;

    t_start = std::chrono::steady_clock::now ();
    t_cpu_start = clock ();
    t_next = t_start;
    n_secs_next_pause = cfg.n_secs_pause;
    if (!enabled ())
        return;

    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize (1 << 30);
    args.SetMaxSendMessageSize (1 << 30);
    args.SetCompressionAlgorithm (GRPC_COMPRESS_NONE);
    args.SetInt (GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    const auto channel = grpc::CreateCustomChannel (
        cfg.branch_server, grpc::InsecureChannelCredentials (), args);
    stub = satact_trace::SATActTraceServer::NewStub (channel);
}

grpc::Status NeuroSATClient::query_branch (
    const CaDiCaL::Solver::SATActSnapshot &snapshot,
    NeuroDecisionData &decision, bool &rpc_attempted) {
    rpc_attempted = false;
    satact_trace::BranchRequest args;
    if (!fill_request (args, cfg, snapshot))
        return grpc::Status (grpc::StatusCode::INVALID_ARGUMENT,
                             "invalid local SAT-ACT snapshot");

    satact_trace::BranchDecision response;
    grpc::ClientContext context;
    context.set_compression_algorithm (GRPC_COMPRESS_NONE);
    context.set_deadline (std::chrono::system_clock::now () +
                          std::chrono::seconds (cfg.timeout_s));
    rpc_attempted = true;
    const grpc::Status status = stub->query_branch (&context, args, &response);
    if (status.ok ())
        copy_decision (response, decision);
    return status;
}

NeuroSATClient::ValidatedDecision NeuroSATClient::validate_decision (
    const CaDiCaL::Solver::SATActSnapshot &snapshot,
    const NeuroDecisionData &decision) const {
    ValidatedDecision result;
    if (!decision.success || decision.model_variant != cfg.model_variant ||
        !std::isfinite (decision.n_secs_inference) ||
        decision.n_secs_inference < 0.0 ||
        !std::isfinite (decision.selected_log_probability) ||
        !decision.action_logits_finite)
        return result;

    const int expected_logits =
        2 * snapshot.n_vars + (variant_allows_defer () ? 1 : 0);
    if (cfg.response_payload == "full") {
        if (decision.response_payload !=
                (int) satact_trace::RESPONSE_PAYLOAD_FULL ||
            decision.action_logits_size != expected_logits)
            return result;
    } else if (decision.response_payload !=
                   (int) satact_trace::RESPONSE_PAYLOAD_COMPACT ||
               decision.action_logits_size) {
        return result;
    }

    if (decision.action == (int) satact_trace::ACTION_DEFER) {
        if (!variant_allows_defer ())
            return result;
        result.kind = DECISION_DEFER;
        return result;
    }
    if (decision.action != (int) satact_trace::ACTION_LITERAL)
        return result;

    const int literal_idx = decision.selected_literal_index;
    if (literal_idx < 0 || literal_idx >= 2 * snapshot.n_vars)
        return result;
    const int internal_idx = literal_idx / 2;
    if (snapshot.internal_to_external.size () !=
            (size_t) snapshot.n_vars ||
        snapshot.candidate_variable.size () != (size_t) snapshot.n_vars ||
        snapshot.assignment_value.size () != (size_t) snapshot.n_vars ||
        !snapshot.candidate_variable[internal_idx] ||
        snapshot.assignment_value[internal_idx] != 0)
        return result;

    const int external_positive =
        snapshot.internal_to_external[internal_idx];
    if (!external_positive)
        return result;
    result.kind = DECISION_LITERAL;
    result.literal = literal_idx % 2 ? -external_positive : external_positive;
    return result;
}

void NeuroSATClient::advance_pause () {
    t_next = std::chrono::steady_clock::now () +
             std::chrono::duration_cast<std::chrono::steady_clock::duration> (
                 std::chrono::duration<double> (n_secs_next_pause));
    n_secs_next_pause *= cfg.n_secs_pause_inc;
}

int NeuroSATClient::satact_decision_lit (CaDiCaL::Solver &solver) {
    if (!enabled () || !stub)
        return 0;
    decision_opportunities++;
    if (!ready_to_call ())
        return 0;

    CaDiCaL::Solver::SATActSnapshot snapshot;
    if (!solver.satact_snapshot (snapshot) || snapshot.n_vars <= 0 ||
        snapshot.clauses.empty ()) {
        preflight_skips++;
        return 0;
    }

    size_t edges = 0;
    bool oversized = false;
    int candidates = 0;
    for (const bool candidate : snapshot.candidate_variable)
        candidates += candidate ? 1 : 0;
    for (const auto &clause : snapshot.clauses) {
        edges += clause.size ();
        oversized = oversized || clause.size () > cfg.max_lclause_size;
        for (const int lit : clause) {
            const int var = lit < 0 ? -lit : lit;
            if (var < 1 || var > snapshot.n_vars) {
                preflight_skips++;
                return 0;
            }
        }
    }
    const size_t cells = 2u * (size_t) snapshot.n_vars +
                         snapshot.clauses.size () + edges;
    if (!candidates || !edges ||
        (!cfg.call_if_too_big &&
         (oversized || cells > cfg.max_n_nodes_cells))) {
        preflight_skips++;
        return 0;
    }

    NeuroDecisionData decision;
    bool rpc_attempted = false;
    const auto wait_start = std::chrono::steady_clock::now ();
    const grpc::Status status = query_branch (snapshot, decision,
                                              rpc_attempted);
    if (!rpc_attempted) {
        preflight_skips++;
        return 0;
    }
    rpc_attempts++;
    n_secs_wait += std::chrono::duration<double> (
        std::chrono::steady_clock::now () - wait_start).count ();
    if (!status.ok () || !decision.success) {
        rpc_errors++;
        return 0;
    }
    const ValidatedDecision validated =
        validate_decision (snapshot, decision);
    if (validated.kind == DECISION_INVALID) {
        invalid_responses++;
        return 0;
    }

    rpc_successes++;
    n_secs_inference += decision.n_secs_inference;
    advance_pause ();
    if (validated.kind == DECISION_DEFER) {
        defer_actions++;
        return 0;
    }
    literal_actions++;
    return validated.literal;
}

void NeuroSATClient::end_solve (int status, const CaDiCaL::Solver &solver) {
    FILE *res = fopen (cfg.neuro_outfile.c_str (), "w");
    if (!res)
        return;
    const char *name = status == 10 ? "SAT" : status == 20 ? "UNSAT"
                                                        : "UNKNOWN";
    const double wall = std::chrono::duration<double> (
        std::chrono::steady_clock::now () - t_start).count ();
    const double cpu = (double) (clock () - t_cpu_start) / CLOCKS_PER_SEC;
    fprintf (
        res,
        "%s %.9f %.9f %.9f %llu %llu %llu %llu %llu %llu %llu %llu "
        "%llu %llu %.9f %ld %ld %ld %ld\n",
        name, cpu, wall, n_secs_wait,
        (unsigned long long) decision_opportunities,
        (unsigned long long) rpc_attempts,
        (unsigned long long) rpc_successes,
        (unsigned long long) rpc_errors,
        (unsigned long long) invalid_responses,
        (unsigned long long) literal_actions,
        (unsigned long long) defer_actions,
        (unsigned long long) preflight_skips,
        (unsigned long long) pause_skips,
        (unsigned long long) budget_skips, n_secs_inference,
        (long) solver.get_statistic_value ("conflicts"),
        (long) solver.get_statistic_value ("decisions"),
        (long) solver.get_statistic_value ("propagations"),
        (long) solver.get_statistic_value ("restarts"));
    fclose (res);
}
