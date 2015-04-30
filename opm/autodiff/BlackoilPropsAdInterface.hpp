/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.

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

#ifndef OPM_BLACKOILPROPSADINTERFACE_HEADER_INCLUDED
#define OPM_BLACKOILPROPSADINTERFACE_HEADER_INCLUDED

#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/core/props/BlackoilPhases.hpp>

namespace Opm
{

    /// This class is intended to present a fluid interface for
    /// three-phase black-oil that is easy to use with the AD-using
    /// simulators.
    ///
    /// Most methods are available in two overloaded versions, one
    /// taking a constant vector and returning the same, and one
    /// taking an AD type and returning the same. Derivatives are not
    /// returned separately by any method, only implicitly with the AD
    /// version of the methods.
    class BlackoilPropsAdInterface
    {
    public:
        /// Virtual destructor for inheritance.
        virtual ~BlackoilPropsAdInterface();

        ////////////////////////////
        //      Rock interface    //
        ////////////////////////////

        /// \return   D, the number of spatial dimensions.
        virtual int numDimensions() const = 0;

        /// \return   N, the number of cells.
        virtual int numCells() const = 0;

        /// \return   Array of N porosity values.
        virtual const double* porosity() const = 0;

        /// \return   Array of ND^2 permeability values.
        ///           The D^2 permeability values for a cell are organized as a matrix,
        ///           which is symmetric (so ordering does not matter).
        virtual const double* permeability() const = 0;


        ////////////////////////////
        //      Fluid interface   //
        ////////////////////////////

        typedef AutoDiffBlock<double> ADB;
        typedef ADB::V V;
        typedef ADB::M M;
        typedef std::vector<int> Cells;

        /// \return   Number of active phases (also the number of components).
        virtual int numPhases() const = 0;

        /// \return   Object describing the active phases.
        virtual PhaseUsage phaseUsage() const = 0;

        // ------ Canonical named indices for each phase ------

        /// Canonical named indices for each phase.
        enum PhaseIndex { Water = BlackoilPhases::Aqua, Oil = BlackoilPhases::Liquid,
                          Gas = BlackoilPhases::Vapour,
                          Aqua = BlackoilPhases::Aqua,
                          Liquid = BlackoilPhases::Liquid,
                          Vapour = BlackoilPhases::Vapour,
                          MaxNumPhases = BlackoilPhases::MaxNumPhases};

        // ------ Density ------

        /// Densities of stock components at surface conditions.
        /// \return Array of 3 density values.
        virtual const double* surfaceDensity(int regionIdx = 0) const = 0;


        // ------ Viscosity ------

        /// Water viscosity.
        /// \param[in]  pw     Array of n water pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n viscosity values.
        virtual
        ADB muWat(const ADB& pw,
                  const ADB& T,
                  const Cells& cells) const = 0;

        /// Oil viscosity.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  rs     Array of n gas solution factor values.
        /// \param[in]  cond   Array of n objects, each specifying which phases are present with non-zero saturation in a cell.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n viscosity values.
        virtual
        ADB muOil(const ADB& po,
                  const ADB& T,
                  const ADB& rs,
                  const std::vector<PhasePresence>& cond,
                  const Cells& cells) const = 0;

        /// Gas viscosity.
        /// \param[in]  pg     Array of n gas pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  rv     Array of n vapor oil/gas ratios.
        /// \param[in]  cond   Array of n objects, each specifying which phases are present with non-zero saturation in a cell.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n viscosity values.
        virtual
        ADB muGas(const ADB& pg,
                  const ADB& T,
                  const ADB& rv,
                  const std::vector<PhasePresence>& cond,
                  const Cells& cells) const = 0;

        // ------ Formation volume factor (b) ------

        /// Water formation volume factor.
        /// \param[in]  pw     Array of n water pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n formation volume factor values.
        virtual
        ADB bWat(const ADB& pw,
                 const ADB& T,
                 const Cells& cells) const = 0;

        /// Oil formation volume factor.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  rs     Array of n gas solution factor values.
        /// \param[in]  cond   Array of n objects, each specifying which phases are present with non-zero saturation in a cell.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n formation volume factor values.
        virtual
        ADB bOil(const ADB& po,
                 const ADB& T,
                 const ADB& rs,
                 const std::vector<PhasePresence>& cond,
                 const Cells& cells) const = 0;

        /// Gas formation volume factor.
        /// \param[in]  pg     Array of n gas pressure values.
        /// \param[in]  T      Array of n temperature values.
        /// \param[in]  rv     Array of n vapor oil/gas ratios.
        /// \param[in]  cond   Array of n objects, each specifying which phases are present with non-zero saturation in a cell.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n formation volume factor values.
        virtual
        ADB bGas(const ADB& pg,
                 const ADB& T,
                 const ADB& rv,
                 const std::vector<PhasePresence>& cond,
                 const Cells& cells) const = 0;

        // ------ Rs bubble point curve ------

        /// Bubble point curve for Rs as function of oil pressure.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n bubble point values for Rs.
        virtual
        ADB rsSat(const ADB& po,
                  const Cells& cells) const = 0;

        /// Bubble point curve for Rs as function of oil pressure.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  so     Array of n oil saturation values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n bubble point values for Rs.
        virtual
        ADB rsSat(const ADB& po,
                  const ADB& so,
                  const Cells& cells) const = 0;

        // ------ Rv condensation curve ------

        /// Condensation curve for Rv as function of oil pressure.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n condensation point values for Rv.
        virtual
        ADB rvSat(const ADB& po,
                  const Cells& cells) const = 0;

        /// Condensation curve for Rv as function of oil pressure.
        /// \param[in]  po     Array of n oil pressure values.
        /// \param[in]  so     Array of n oil saturation values.
        /// \param[in]  cells  Array of n cell indices to be associated with the pressure values.
        /// \return            Array of n condensation point values for Rv.
        virtual
        ADB rvSat(const ADB& po,
                  const ADB& so,
                  const Cells& cells) const = 0;

        // ------ Relative permeability ------

        /// Relative permeabilities for all phases.
        /// \param[in]  sw     Array of n water saturation values.
        /// \param[in]  so     Array of n oil saturation values.
        /// \param[in]  sg     Array of n gas saturation values.
        /// \param[in]  cells  Array of n cell indices to be associated with the saturation values.
        /// \return            An std::vector with 3 elements, each an array of n relperm values,
        ///                    containing krw, kro, krg. Use PhaseIndex for indexing into the result.
        virtual
        std::vector<ADB> relperm(const ADB& sw,
                                 const ADB& so,
                                 const ADB& sg,
                                 const Cells& cells) const = 0;


        /// Capillary pressure for all phases.
        /// \param[in]  sw     Array of n water saturation values.
        /// \param[in]  so     Array of n oil saturation values.
        /// \param[in]  sg     Array of n gas saturation values.
        /// \param[in]  cells  Array of n cell indices to be associated with the saturation values.
        /// \return            An std::vector with 3 elements, each an array of n capillary pressure values,
        ///                    containing the offsets for each p_g, p_o, p_w. The capillary pressure between
        ///                    two arbitrary phases alpha and beta is then given as p_alpha - p_beta.
        virtual
        std::vector<ADB> capPress(const ADB& sw,
                                  const ADB& so,
                                  const ADB& sg,
                                  const Cells& cells) const = 0;
                                  
        /// Saturation update for hysteresis behavior.
        /// \param[in]  cells       Array of n cell indices to be associated with the saturation values.
        virtual
        void updateSatHyst(const std::vector<double>& saturation,
                           const std::vector<int>& cells) = 0;

        /// Update for max oil saturation.
        virtual                  
        void updateSatOilMax(const std::vector<double>& saturation) = 0;
    };

} // namespace Opm

#endif // OPM_BLACKOILPROPSADINTERFACE_HEADER_INCLUDED
