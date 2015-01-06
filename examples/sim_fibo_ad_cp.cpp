/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
  Copyright 2014 Dr. Blatt - HPC-Simulation-Software & Services

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/


#if HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <dune/common/version.hh>

#include <opm/core/utility/platform_dependent/disable_warnings.h>

#if DUNE_VERSION_NEWER(DUNE_COMMON, 2, 3)
#include <dune/common/parallel/mpihelper.hh>
#else
#include <dune/common/mpihelper.hh>
#endif
#include <dune/grid/CpGrid.hpp>
#include <dune/grid/common/GridAdapter.hpp>

#include <opm/core/utility/platform_dependent/reenable_warnings.h>

#include <opm/core/pressure/FlowBCManager.hpp>

#include <opm/core/grid.h>
#include <opm/core/grid/cornerpoint_grid.h>
#include <opm/autodiff/GridHelpers.hpp>

#include <opm/core/wells.h>
#include <opm/core/wells/WellsManager.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/simulator/initState.hpp>
#include <opm/core/simulator/initStateEquil.hpp>
#include <opm/core/simulator/SimulatorReport.hpp>
#include <opm/core/simulator/SimulatorTimer.hpp>
#include <opm/core/utility/miscUtilities.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/utility/thresholdPressures.hpp> // Note: the GridHelpers must be included before this (to make overloads available). \TODO: Fix.

#include <opm/core/io/eclipse/EclipseWriter.hpp>
#include <opm/core/props/BlackoilPropertiesBasic.hpp>
#include <opm/core/props/BlackoilPropertiesFromDeck.hpp>
#include <opm/core/props/rock/RockCompressibility.hpp>

#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/autodiff/NewtonIterationBlackoilSimple.hpp>
#include <opm/autodiff/NewtonIterationBlackoilCPR.hpp>

#include <opm/core/simulator/BlackoilState.hpp>
#include <opm/autodiff/WellStateFullyImplicitBlackoil.hpp>

#include <opm/autodiff/SimulatorFullyImplicitBlackoil.hpp>
#include <opm/autodiff/BlackoilPropsAdFromDeck.hpp>
#include <opm/core/utility/share_obj.hpp>

#include <opm/parser/eclipse/Deck/Deck.hpp>
#include <opm/parser/eclipse/Parser/Parser.hpp>
#include <opm/parser/eclipse/EclipseState/checkDeck.hpp>
#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <memory>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>

namespace
{
    void warnIfUnusedParams(const Opm::parameter::ParameterGroup& param)
    {
        if (param.anyUnused()) {
            std::cout << "--------------------   Unused parameters:   --------------------\n";
            param.displayUsage();
            std::cout << "----------------------------------------------------------------" << std::endl;
        }
    }
} // anon namespace



// ----------------- Main program -----------------
int
main(int argc, char** argv)
try
{
    // Must ensure an instance of the helper is created to initialise MPI,
    // but we don't use the helper here.
    // Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
    Dune::MPIHelper::instance(argc, argv);
    using namespace Opm;

    std::cout << "\n================    Test program for fully implicit three-phase black-oil flow     ===============\n\n";
    parameter::ParameterGroup param(argc, argv, false);
    std::cout << "---------------    Reading parameters     ---------------" << std::endl;

    // If we have a "deck_filename", grid and props will be read from that.
    bool use_deck = param.has("deck_filename");
    if (!use_deck) {
        OPM_THROW(std::runtime_error, "This program must be run with an input deck. "
                  "Specify the deck with deck_filename=deckname.data (for example).");
    }
    std::shared_ptr<Dune::CpGrid> grid;
    std::shared_ptr<BlackoilPropertiesInterface> props;
    std::shared_ptr<BlackoilPropsAdInterface> new_props;
    std::shared_ptr<RockCompressibility> rock_comp;
    BlackoilState state;
    // bool check_well_controls = false;
    // int max_well_control_iterations = 0;
    double gravity[3] = { 0.0 };
    std::string deck_filename = param.get<std::string>("deck_filename");

    Opm::ParserPtr parser(new Opm::Parser() );
    Opm::LoggerPtr logger(new Opm::Logger());
    bool strict_parsing = param.getDefault("strict_parsing", true);
    Opm::DeckConstPtr deck;
    std::shared_ptr<EclipseState> eclipseState;
    try {
        deck = parser->parseFile(deck_filename, logger);
        checkDeck(deck, logger);
        eclipseState.reset(new EclipseState(deck, logger));
    }
    catch (const std::invalid_argument& e) {
        if (logger->size() > 0) {
            std::cerr << "Issues found while parsing the deck file:\n";
            logger->printAll(std::cerr);
        }
        std::cerr << "error while parsing the deck file: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    if (logger->size() > 0) {
        std::cerr << "Issues found while parsing the deck file:\n";
        logger->printAll(std::cerr);
    }

    // Grid init
    grid.reset(new Dune::CpGrid());
    std::vector<double> porv = eclipseState->getDoubleGridProperty("PORV")->getData();
    grid->processEclipseFormat(deck, false, false, false, porv);

    const PhaseUsage pu = Opm::phaseUsageFromDeck(deck);
    Opm::EclipseWriter outputWriter(param, eclipseState, pu,
                                    Opm::UgGridHelpers::numCells(*grid),
                                    Opm::UgGridHelpers::globalCell(*grid));

    // Rock and fluid init
    props.reset(new BlackoilPropertiesFromDeck(deck, eclipseState,
                                               Opm::UgGridHelpers::numCells(*grid),
                                               Opm::UgGridHelpers::globalCell(*grid),
                                               Opm::UgGridHelpers::cartDims(*grid),
                                               Opm::UgGridHelpers::beginCellCentroids(*grid),
                                               Opm::UgGridHelpers::dimensions(*grid), param));
    new_props.reset(new BlackoilPropsAdFromDeck(deck, eclipseState, *grid));
    // check_well_controls = param.getDefault("check_well_controls", false);
    // max_well_control_iterations = param.getDefault("max_well_control_iterations", 10);
    // Rock compressibility.
    rock_comp.reset(new RockCompressibility(deck, eclipseState));

    // Gravity.
    gravity[2] = deck->hasKeyword("NOGRAV") ? 0.0 : unit::gravity;

    // Init state variables (saturation and pressure).
    if (param.has("init_saturation")) {
        initStateBasic(grid->numCells(), &(grid->globalCell())[0],
                       &(grid->logicalCartesianSize()[0]),
                       grid->numFaces(), UgGridHelpers::faceCells(*grid),
                       grid->beginFaceCentroids(),
                       grid->beginCellCentroids(), Dune::CpGrid::dimension,
                       *props, param, gravity[2], state);
        initBlackoilSurfvol(grid->numCells(), *props, state);
        enum { Oil = BlackoilPhases::Liquid, Gas = BlackoilPhases::Vapour };
        if (pu.phase_used[Oil] && pu.phase_used[Gas]) {
            const int np = props->numPhases();
            const int nc = grid->numCells();
            for (int c = 0; c < nc; ++c) {
                state.gasoilratio()[c] = state.surfacevol()[c*np + pu.phase_pos[Gas]]
                    / state.surfacevol()[c*np + pu.phase_pos[Oil]];
            }
        }
    } else if (deck->hasKeyword("EQUIL") && props->numPhases() == 3) {
        OPM_THROW(std::logic_error, "sim_fibo_ad_cp does not support EQUIL initialization.");
    } else {
        initBlackoilStateFromDeck(grid->numCells(), &(grid->globalCell())[0],
                                  grid->numFaces(), UgGridHelpers::faceCells(*grid),
                                  grid->beginFaceCentroids(),
                                  grid->beginCellCentroids(), Dune::CpGrid::dimension,
                                  *props, deck, gravity[2], state);
    }

    bool use_gravity = (gravity[0] != 0.0 || gravity[1] != 0.0 || gravity[2] != 0.0);
    const double *grav = use_gravity ? &gravity[0] : 0;

    // Solver for Newton iterations.
    std::unique_ptr<NewtonIterationBlackoilInterface> fis_solver;
    if (param.getDefault("use_cpr", true)) {
        fis_solver.reset(new NewtonIterationBlackoilCPR(param));
    } else {
        fis_solver.reset(new NewtonIterationBlackoilSimple(param));
    }

    // Write parameters used for later reference.
    bool output = param.getDefault("output", true);
    std::string output_dir;
    if (output) {
        // Create output directory if needed.
        output_dir =
            param.getDefault("output_dir", std::string("output"));
        boost::filesystem::path fpath(output_dir);
        try {
            create_directories(fpath);
        }
        catch (...) {
            OPM_THROW(std::runtime_error, "Creating directories failed: " << fpath);
        }
        // Write simulation parameters.
        param.writeParam(output_dir + "/simulation.param");
    }

    Opm::TimeMapConstPtr timeMap(eclipseState->getSchedule()->getTimeMap());
    SimulatorTimer simtimer;

    // initialize variables
    simtimer.init(timeMap);

    Opm::DerivedGeology geology(*grid, *new_props, eclipseState, false, grav);

    std::vector<double> threshold_pressures = thresholdPressures(deck, eclipseState, *grid);

    SimulatorFullyImplicitBlackoil<Dune::CpGrid> simulator(param,
                                                           *grid,
                                                           geology,
                                                           *new_props,
                                                           rock_comp->isActive() ? rock_comp.get() : 0,
                                                           *fis_solver,
                                                           grav,
                                                           deck->hasKeyword("DISGAS"),
                                                           deck->hasKeyword("VAPOIL"),
                                                           eclipseState,
                                                           outputWriter,
                                                           threshold_pressures);

    std::cout << "\n\n================ Starting main simulation loop ===============\n"
              << std::flush;

    SimulatorReport fullReport = simulator.run(simtimer, state);

    std::cout << "\n\n================    End of simulation     ===============\n\n";
    fullReport.report(std::cout);

    if (output) {
        std::string filename = output_dir + "/walltime.txt";
        std::fstream tot_os(filename.c_str(),std::fstream::trunc | std::fstream::out);
        fullReport.reportParam(tot_os);
        warnIfUnusedParams(param);
    }
}
catch (const std::exception &e) {
    std::cerr << "Program threw an exception: " << e.what() << "\n";
    throw;
}

