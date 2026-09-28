#ifndef MY_PROPAGATOR_V7_HPP
#define MY_PROPAGATOR_V7_HPP

#include "cadical.hpp"
#include "neuro_client.hpp"
#include <vector>

class MinimalPropagator : public CaDiCaL::ExternalPropagator {
public:
    void notify_assignment (const std::vector<int> &lits) override;
    void notify_new_decision_level () override;
    void notify_backtrack (size_t new_level) override;
    bool cb_check_found_model (const std::vector<int> &model) override;
    bool cb_has_external_clause (bool &is_forgettable) override;
    int cb_add_external_clause_lit () override;
};

class MyDecisionPropagator : public MinimalPropagator {
    CaDiCaL::Solver *solver;
    NeuroSATClient *neuro;

public:
    MyDecisionPropagator (CaDiCaL::Solver *solver,
                          NeuroSATClient *neuro);
    int cb_decide () override;
};

class OracleFirstDecisionPropagator : public MinimalPropagator {
    CaDiCaL::Solver *solver;
    int lit;
    bool use_default_phase;
    bool used = false;
    int chosen_lit = 0;
    std::vector<int> assignment_stack;
    std::vector<size_t> level_sizes;

    int oracle_var () const;
    bool oracle_var_assigned () const;

public:
    OracleFirstDecisionPropagator (CaDiCaL::Solver *solver, int oracle_lit,
                                   bool default_phase);
    void notify_assignment (const std::vector<int> &lits) override;
    void notify_new_decision_level () override;
    void notify_backtrack (size_t new_level) override;
    int cb_decide () override;
    int decision_lit () const;
};

#endif
