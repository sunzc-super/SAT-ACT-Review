#include "cadical.hpp"
#include "my_propagator.hpp"
#include "neuro_client.hpp"
#include <chrono>
#include <cctype>
#include <cmath>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct ParsedInput {
    int max_var = 0;
    std::vector<std::vector<int>> clauses;
};

struct RunTiming {
    double wall = 0.0;
    double cpu = 0.0;
};

static volatile std::sig_atomic_t termination_requested = 0;

static void request_termination (int) { termination_requested = 1; }

class TimeoutTerminator : public CaDiCaL::Terminator {
    bool active = false;
    std::chrono::steady_clock::time_point deadline;

public:
    void set (int seconds) {
        active = seconds > 0;
        deadline = std::chrono::steady_clock::now () +
                   std::chrono::seconds (seconds);
    }

    bool terminate () override {
        return termination_requested ||
               (active && std::chrono::steady_clock::now () >= deadline);
    }
};

static unsigned parse_positive_unsigned (const char *value,
                                         const char *option) {
    const std::string text (value);
    if (text.empty ())
        throw std::runtime_error (std::string ("invalid ") + option);
    for (char ch : text)
        if (!std::isdigit ((unsigned char) ch))
            throw std::runtime_error (std::string ("invalid ") + option);
    const unsigned long parsed = std::stoul (text);
    if (parsed == 0 || parsed > UINT_MAX)
        throw std::runtime_error (std::string ("invalid ") + option);
    return (unsigned) parsed;
}

static ParsedInput read_dimacs (const std::string &path) {
    std::ifstream in (path.c_str ());
    if (!in)
        throw std::runtime_error ("can not open DIMACS file '" + path + "'");

    ParsedInput parsed;
    std::vector<int> clause;
    std::string token;
    while (in >> token) {
        if (token == "c") {
            std::string rest;
            std::getline (in, rest);
            continue;
        }
        if (token == "p") {
            std::string format;
            int n_clauses;
            in >> format >> parsed.max_var >> n_clauses;
            (void) n_clauses;
            continue;
        }
        const int lit = std::atoi (token.c_str ());
        if (lit == 0) {
            parsed.clauses.push_back (clause);
            clause.clear ();
        } else {
            const int var = lit < 0 ? -lit : lit;
            if (var > parsed.max_var)
                parsed.max_var = var;
            clause.push_back (lit);
        }
    }
    if (!clause.empty ())
        parsed.clauses.push_back (clause);
    return parsed;
}

static void release_clauses (ParsedInput &input) {
    std::vector<std::vector<int>> ().swap (input.clauses);
}

static const char *status_name (int res) {
    return res == 10 ? "SAT" : res == 20 ? "UNSAT" : "UNKNOWN";
}

static void print_status (int res) {
    if (res == 10)
        std::cout << "s SATISFIABLE\n";
    else if (res == 20)
        std::cout << "s UNSATISFIABLE\n";
    else
        std::cout << "s UNKNOWN\n";
}

static int status_exit_code (int res) {
    return res == 10 ? 10 : res == 20 ? 20 : 0;
}

static void add_formula (CaDiCaL::Solver &solver, const ParsedInput &input) {
    solver.declare_more_variables (input.max_var);
    for (const auto &clause : input.clauses) {
        for (int lit : clause)
            solver.add (lit);
        solver.add (0);
    }
}

static void write_result_file (const std::string &path, int status,
                               const CaDiCaL::Solver &solver,
                               const RunTiming &timing) {
    FILE *res = fopen (path.c_str (), "w");
    if (!res)
        return;
    fprintf (res,
             "%s %.9f %.9f %.9f 0 0 0 0 0 0 0 0 0 0 %.9f %ld %ld %ld %ld\n",
             status_name (status), timing.cpu, timing.wall, 0.0, 0.0,
             (long) solver.get_statistic_value ("conflicts"),
             (long) solver.get_statistic_value ("decisions"),
             (long) solver.get_statistic_value ("propagations"),
             (long) solver.get_statistic_value ("restarts"));
    fclose (res);
}

static bool is_neuro_mode (const std::string &mode) {
    return mode == "SATACT";
}

static bool is_satact_model_variant (const std::string &variant) {
    return variant == "satact";
}

static void usage (const char *name) {
    std::cerr
        << "usage: " << name << " [neuro-options] input.cnf\n"
        << "  --mode NONE|CADICAL-BASELINE|SATACT|LOOKAHEAD|ORACLE_LOOKAHEAD|ORACLE_LOOKAHEAD_VAR\n"
        << "  --model_variant satact\n"
        << "  --decide_strategy FIRST\n"
        << "  --branch_server HOST:PORT\n"
        << "  --timeout_s N\n"
        << "  --n_secs_pause X\n"
        << "  --n_secs_pause_inc X\n"
        << "  --max_lclause_size N\n"
        << "  --max_n_nodes_cells N\n"
        << "  --call_if_too_big\n"
        << "  --neuro_outfile PATH\n"
        << "  --neuro_calls K\n"
        << "  --response_payload compact|full\n"
        << "  -c N    conflict limit\n"
        << "  -d N    decision limit\n"
        << "  -t N    real-time limit in seconds\n"
        << "  --plain disable preprocessing options\n"
        << "  -q, --quiet, --quite\n"
        << "  -v\n"
        << "  any CaDiCaL --<option>, --<option>=<value>, or --no-<option>\n";
}

int main (int argc, char **argv) {
    std::signal (SIGTERM, request_termination);
    std::signal (SIGINT, request_termination);

    NeuroSATConfig ncfg;
    std::string dimacs_path;
    std::vector<std::string> solver_options;
    std::vector<std::string> solver_configurations;
    int verbose_increments = 0;
    bool quiet = false;
    int conflict_limit = -1;
    int decision_limit = -1;
    int time_limit = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&] () -> const char * {
            if (++i == argc)
                throw std::runtime_error ("missing value for " + arg);
            return argv[i];
        };

        if (arg == "--mode")
            ncfg.mode = need_value ();
        else if (arg == "--model_variant")
            ncfg.model_variant = need_value ();
        else if (arg == "--decide_strategy")
            ncfg.decide_strategy = need_value ();
        else if (arg == "--branch_server")
            ncfg.branch_server = need_value ();
        else if (arg == "--timeout_s")
            ncfg.timeout_s = (unsigned) std::stoul (need_value ());
        else if (arg == "--n_secs_pause")
            ncfg.n_secs_pause = std::stod (need_value ());
        else if (arg == "--n_secs_pause_inc")
            ncfg.n_secs_pause_inc = std::stod (need_value ());
        else if (arg == "--max_lclause_size")
            ncfg.max_lclause_size = (unsigned) std::stoul (need_value ());
        else if (arg == "--max_n_nodes_cells")
            ncfg.max_n_nodes_cells = (unsigned) std::stoul (need_value ());
        else if (arg == "--call_if_too_big")
            ncfg.call_if_too_big = true;
        else if (arg == "--neuro_outfile")
            ncfg.neuro_outfile = need_value ();
        else if (arg == "--neuro_calls")
            ncfg.neuro_calls =
                parse_positive_unsigned (need_value (), "neuro_calls");
        else if (arg == "--response_payload")
            ncfg.response_payload = need_value ();
        else if (arg == "-c") {
            conflict_limit = std::stoi (need_value ());
            if (conflict_limit < 0)
                throw std::runtime_error ("invalid conflict limit");
        } else if (arg == "-d") {
            decision_limit = std::stoi (need_value ());
            if (decision_limit < 0)
                throw std::runtime_error ("invalid decision limit");
        } else if (arg == "-t") {
            time_limit = std::stoi (need_value ());
            if (time_limit < 0)
                throw std::runtime_error ("invalid time limit");
        } else if (arg == "-q" || arg == "--quiet" || arg == "--quite")
            quiet = true;
        else if (arg == "-v")
            verbose_increments++;
        else if (arg == "-h" || arg == "--help") {
            usage (argv[0]);
            return 0;
        } else if (!arg.empty () && arg[0] == '-') {
            if (arg.size () > 2 && arg.substr (0, 2) == "--" &&
                CaDiCaL::Solver::is_valid_configuration (arg.c_str () + 2))
                solver_configurations.push_back (arg.substr (2));
            else if (CaDiCaL::Solver::is_valid_long_option (arg.c_str ()))
                solver_options.push_back (arg);
            else
                throw std::runtime_error ("unknown option '" + arg + "'");
        } else {
            dimacs_path = arg;
        }
    }

    if (dimacs_path.empty ()) {
        usage (argv[0]);
        return 1;
    }
    if (ncfg.mode != "NONE" && ncfg.mode != "CADICAL-BASELINE" &&
        !is_neuro_mode (ncfg.mode) && ncfg.mode != "LOOKAHEAD" &&
        ncfg.mode != "ORACLE_LOOKAHEAD" &&
        ncfg.mode != "ORACLE_LOOKAHEAD_VAR")
        throw std::runtime_error ("unknown mode '" + ncfg.mode + "'");
    if (is_neuro_mode (ncfg.mode) &&
        !is_satact_model_variant (ncfg.model_variant))
        throw std::runtime_error ("invalid SAT-ACT model variant '" +
                                  ncfg.model_variant + "'");
    if (ncfg.decide_strategy != "FIRST")
        throw std::runtime_error ("unknown decide_strategy '" +
                                  ncfg.decide_strategy + "'");
    if (!std::isfinite (ncfg.n_secs_pause) || ncfg.n_secs_pause < 0.0)
        throw std::runtime_error ("n_secs_pause must be finite and >= 0");
    if (!std::isfinite (ncfg.n_secs_pause_inc) ||
        ncfg.n_secs_pause_inc <= 0.0)
        throw std::runtime_error (
            "n_secs_pause_inc must be finite and > 0");
    if (ncfg.response_payload != "compact" &&
        ncfg.response_payload != "full")
        throw std::runtime_error ("unknown response_payload '" +
                                  ncfg.response_payload + "'");

    ParsedInput input = read_dimacs (dimacs_path);
    const int max_var = input.max_var;

    auto configure_solver = [&] (CaDiCaL::Solver &solver,
                                 bool clean_output) {
        for (const auto &config : solver_configurations)
            if (!solver.configure (config.c_str ()))
                throw std::runtime_error ("invalid solver configuration '" +
                                          config + "'");
        solver.set ("report", clean_output ? 0 : 1);
        if (quiet || clean_output)
            solver.set ("quiet", 1);
        for (int i = 0; i < verbose_increments; ++i)
            solver.set ("verbose", solver.get ("verbose") + 1);
        for (const auto &opt : solver_options)
            if (!solver.set_long_option (opt.c_str ()))
                throw std::runtime_error ("invalid CaDiCaL option '" + opt +
                                          "'");
        if (conflict_limit >= 0)
            solver.limit ("conflicts", conflict_limit);
        if (decision_limit >= 0)
            solver.limit ("decisions", decision_limit);
    };

    auto connect_terminator = [&] (CaDiCaL::Solver &solver,
                                   TimeoutTerminator &terminator) {
        terminator.set (time_limit);
        solver.connect_terminator (&terminator);
    };

    auto disconnect_terminator = [&] (CaDiCaL::Solver &solver) {
        solver.disconnect_terminator ();
    };

    auto solve_with_timing = [&] (CaDiCaL::Solver &solver,
                                  RunTiming &timing) {
        TimeoutTerminator terminator;
        terminator.set (time_limit);
        solver.connect_terminator (&terminator);
        const auto t_start = std::chrono::steady_clock::now ();
        const clock_t cpu_start = clock ();
        const int res = solver.solve ();
        timing.wall = std::chrono::duration<double> (
            std::chrono::steady_clock::now () - t_start).count ();
        timing.cpu = (double) (clock () - cpu_start) / CLOCKS_PER_SEC;
        solver.disconnect_terminator ();
        return res;
    };

    if (ncfg.mode == "LOOKAHEAD") {
        CaDiCaL::Solver solver;
        configure_solver (solver, true);
        add_formula (solver, input);
        release_clauses (input);
        TimeoutTerminator terminator;
        connect_terminator (solver, terminator);
        const int lit = solver.lookahead ();
        disconnect_terminator (solver);
        std::cout << lit << "\n";
        return 0;
    }

    if (ncfg.mode == "ORACLE_LOOKAHEAD" ||
        ncfg.mode == "ORACLE_LOOKAHEAD_VAR") {
        const bool use_default_phase = ncfg.mode == "ORACLE_LOOKAHEAD_VAR";
        CaDiCaL::Solver lookahead_solver;
        CaDiCaL::Solver oracle_solver;
        configure_solver (lookahead_solver, true);
        configure_solver (oracle_solver, false);
        add_formula (lookahead_solver, input);
        add_formula (oracle_solver, input);
        release_clauses (input);

        TimeoutTerminator lookahead_terminator;
        connect_terminator (lookahead_solver, lookahead_terminator);
        const int oracle_lit = lookahead_solver.lookahead ();
        disconnect_terminator (lookahead_solver);

        OracleFirstDecisionPropagator oracle_propagator (
            &oracle_solver, oracle_lit, use_default_phase);
        if (oracle_lit) {
            oracle_solver.connect_external_propagator (&oracle_propagator);
            oracle_solver.add_observed_var (
                oracle_lit < 0 ? -oracle_lit : oracle_lit);
        }
        RunTiming oracle_timing;
        const int oracle_res =
            solve_with_timing (oracle_solver, oracle_timing);
        if (oracle_lit)
            oracle_solver.disconnect_external_propagator ();

        std::cout << "c oracle lookahead lit " << oracle_lit << "\n";
        if (use_default_phase) {
            std::cout << "c oracle lookahead var "
                      << (oracle_lit < 0 ? -oracle_lit : oracle_lit) << "\n";
            std::cout << "c oracle default phase lit "
                      << oracle_propagator.decision_lit () << "\n";
        }
        print_status (oracle_res);
        oracle_solver.statistics ();
        oracle_solver.resources ();
        write_result_file (ncfg.neuro_outfile, oracle_res, oracle_solver,
                           oracle_timing);
        return status_exit_code (oracle_res);
    }

    CaDiCaL::Solver solver;
    configure_solver (solver, false);
    add_formula (solver, input);
    release_clauses (input);

    NeuroSATClient neuro (ncfg);
    neuro.begin_solve ();
    const bool use_external =
        neuro.enabled () || ncfg.mode == "CADICAL-BASELINE";
    int res;
    RunTiming timing;
    if (use_external) {
        MyDecisionPropagator propagator (&solver, &neuro);
        solver.connect_external_propagator (&propagator);
        for (int var = 1; var <= max_var; ++var)
            solver.add_observed_var (var);
        res = solve_with_timing (solver, timing);
        solver.disconnect_external_propagator ();
    } else {
        res = solve_with_timing (solver, timing);
    }
    neuro.end_solve (res, solver);

    print_status (res);
    solver.statistics ();
    solver.resources ();
    return status_exit_code (res);
}
