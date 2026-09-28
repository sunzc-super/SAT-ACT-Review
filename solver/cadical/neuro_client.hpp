#ifndef SATACT_NEURO_CLIENT_HPP
#define SATACT_NEURO_CLIENT_HPP

#include "cadical.hpp"
#include "satact_trace.grpc.pb.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <string>

struct NeuroSATConfig {
    std::string mode = "NONE";
    std::string model_variant = "satact";
    std::string decide_strategy = "FIRST";
    std::string branch_server = "localhost:41070";
    unsigned timeout_s = 120;
    double n_secs_pause = 0.0;
    double n_secs_pause_inc = 1.0;
    unsigned max_lclause_size = UINT32_MAX;
    unsigned max_n_nodes_cells = UINT32_MAX;
    bool call_if_too_big = false;
    std::string neuro_outfile = "out";
    unsigned neuro_calls = 1;
    std::string response_payload = "compact";
};

struct NeuroDecisionData {
    bool success = false;
    double n_secs_inference = 0.0;
    std::string model_variant;
    int action = 0;
    int selected_literal_index = -1;
    double selected_log_probability = 0.0;
    int response_payload = 0;
    int action_logits_size = 0;
    bool action_logits_finite = true;
};

class NeuroSATClient {
    enum DecisionKind { DECISION_INVALID, DECISION_LITERAL, DECISION_DEFER };

    struct ValidatedDecision {
        DecisionKind kind = DECISION_INVALID;
        int literal = 0;
    };

    NeuroSATConfig cfg;
    std::unique_ptr<satact_trace::SATActTraceServer::Stub> stub;

    uint64_t decision_opportunities = 0;
    uint64_t rpc_attempts = 0;
    uint64_t rpc_successes = 0;
    uint64_t rpc_errors = 0;
    uint64_t invalid_responses = 0;
    uint64_t literal_actions = 0;
    uint64_t defer_actions = 0;
    uint64_t preflight_skips = 0;
    uint64_t pause_skips = 0;
    uint64_t budget_skips = 0;
    double n_secs_inference = 0.0;
    double n_secs_wait = 0.0;
    double n_secs_next_pause = 0.0;
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_next;
    clock_t t_cpu_start{};

    bool ready_to_call ();
    bool variant_allows_defer () const;
    grpc::Status query_branch (
        const CaDiCaL::Solver::SATActSnapshot &snapshot,
        NeuroDecisionData &decision, bool &rpc_attempted);
    ValidatedDecision validate_decision (
        const CaDiCaL::Solver::SATActSnapshot &snapshot,
        const NeuroDecisionData &decision) const;
    void advance_pause ();

public:
    explicit NeuroSATClient (const NeuroSATConfig &config);
    bool enabled () const;
    void begin_solve ();
    int satact_decision_lit (CaDiCaL::Solver &solver);
    void end_solve (int status, const CaDiCaL::Solver &solver);
};

#endif
