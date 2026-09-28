#include "my_propagator.hpp"

void MinimalPropagator::notify_assignment (const std::vector<int> &lits) {
    (void) lits;
}

void MinimalPropagator::notify_new_decision_level () {}

void MinimalPropagator::notify_backtrack (size_t new_level) {
    (void) new_level;
}

bool MinimalPropagator::cb_check_found_model (const std::vector<int> &model) {
    (void) model;
    return true;
}

bool MinimalPropagator::cb_has_external_clause (bool &is_forgettable) {
    (void) is_forgettable;
    return false;
}

int MinimalPropagator::cb_add_external_clause_lit () { return 0; }

MyDecisionPropagator::MyDecisionPropagator (CaDiCaL::Solver *solver_arg,
                                            NeuroSATClient *neuro_arg)
    : solver (solver_arg), neuro (neuro_arg) {}

int MyDecisionPropagator::cb_decide () {
    if (!neuro || !neuro->enabled ())
        return 0;
    return neuro->satact_decision_lit (*solver);
}

OracleFirstDecisionPropagator::OracleFirstDecisionPropagator (
    CaDiCaL::Solver *solver_arg, int oracle_lit, bool default_phase)
    : solver (solver_arg), lit (oracle_lit),
      use_default_phase (default_phase) {}

int OracleFirstDecisionPropagator::oracle_var () const {
    return lit < 0 ? -lit : lit;
}

bool OracleFirstDecisionPropagator::oracle_var_assigned () const {
    const int var = oracle_var ();
    for (int assigned : assignment_stack) {
        const int assigned_var = assigned < 0 ? -assigned : assigned;
        if (assigned_var == var)
            return true;
    }
    return false;
}

void OracleFirstDecisionPropagator::notify_assignment (
    const std::vector<int> &lits) {
    for (int assigned : lits)
        assignment_stack.push_back (assigned);
}

void OracleFirstDecisionPropagator::notify_new_decision_level () {
    level_sizes.push_back (assignment_stack.size ());
}

void OracleFirstDecisionPropagator::notify_backtrack (size_t new_level) {
    const size_t keep = level_sizes.empty () ? assignment_stack.size ()
                                             : level_sizes[new_level];
    while (assignment_stack.size () > keep)
        assignment_stack.pop_back ();
    while (level_sizes.size () > new_level)
        level_sizes.pop_back ();
}

int OracleFirstDecisionPropagator::cb_decide () {
    if (used || !lit || oracle_var_assigned ())
        return 0;
    used = true;
    chosen_lit = use_default_phase ? solver->default_decision_lit (oracle_var ())
                                   : lit;
    return chosen_lit;
}

int OracleFirstDecisionPropagator::decision_lit () const {
    return chosen_lit;
}
