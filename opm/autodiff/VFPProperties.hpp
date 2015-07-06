/*
  Copyright 2015 SINTEF ICT, Applied Mathematics.

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

#ifndef OPM_AUTODIFF_VFPPROPERTIES_HPP_
#define OPM_AUTODIFF_VFPPROPERTIES_HPP_

#include <opm/core/wells.h>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/parser/eclipse/EclipseState/Tables/VFPProdTable.hpp>
#include <opm/parser/eclipse/EclipseState/Tables/VFPInjTable.hpp>
#include <boost/multi_array.hpp>

#include <map>

namespace Opm {

/**
 * Class which linearly interpolates BHP as a function of rate, tubing head pressure,
 * water fraction, gas fraction, and artificial lift for production VFP tables, and similarly
 * the BHP as a function of the rate and tubing head pressure.
 */
class VFPProperties {
public:
    typedef AutoDiffBlock<double> ADB;

    /**
     * Empty constructor
     */
    VFPProperties();

    /**
     * Constructor
     * Takes *no* ownership of data.
     * @param inj_table  A *single* VFPINJ table or NULL (no table)
     * @param prod_table A *single* VFPPROD table or NULL (no table)
     */
    VFPProperties(const VFPInjTable* inj_table, const VFPProdTable* prod_table);

    /**
     * Constructor
     * Takes *no* ownership of data.
     * @param inj_tables A map of different VFPINJ tables.
     * @param prod_tables A map of different VFPPROD tables.
     */
    VFPProperties(const std::map<int, VFPInjTable>& inj_tables,
                  const std::map<int, VFPProdTable>& prod_tables);

    /**
     * Constructor
     * Takes *no* ownership of data.
     * @param inj_tables A map of different VFPINJ tables.
     */
    VFPProperties(const std::map<int, VFPInjTable>& inj_tables);

    /**
     * Constructor
     * Takes *no* ownership of data.
     * @param prod_tables A map of different VFPPROD tables.
     */
    VFPProperties(const std::map<int, VFPProdTable>& prod_tables);

    /**
     * Linear interpolation of bhp as function of the input parameters.
     * @param table Table number to use
     * @param wells Wells structure with information about wells in qs
     * @param qs Flow quantities
     * @param thp Tubing head pressure
     * @param alq Artificial lift or other parameter
     *
     * @return The bottom hole pressure, interpolated/extrapolated linearly using
     * the above parameters from the values in the input table.
     */
    ADB prod_bhp(int table,
            const Wells& wells,
            const ADB& qs,
            const ADB& thp,
            const ADB& alq);

    /**
     * Linear interpolation of bhp as a function of the input parameters
     * @param table Table number to use
     * @param flo Production rate of oil, gas or liquid
     * @param thp Tubing head pressure
     * @param wfr Water-oil ratio, water cut, or water-gas ratio
     * @param gfr Gas-oil ratio, gas-liquid ratio, or oil-gas ratio
     * @param alq Artificial lift or other parameter
     *
     * @return The bottom hole pressure, interpolated/extrapolated linearly using
     * the above parameters from the values in the input table.
     */
    double prod_bhp(int table,
            const double& flo,
            const double& thp,
            const double& wfr,
            const double& gfr,
            const double& alq);

    /**
     * Linear interpolation of bhp as a function of the input parameters given as ADBs
     * @param table Table number to use
     * @param flo Production rate of oil, gas or liquid
     * @param thp Tubing head pressure
     * @param wfr Water-oil ratio, water cut, or water-gas ratio
     * @param gfr Gas-oil ratio, gas-liquid ratio, or oil-gas ratio
     * @param alq Artificial lift or other parameter
     *
     * @return The bottom hole pressure, interpolated/extrapolated linearly using
     * the above parameters from the values in the input table, for each entry in the
     * input ADB objects.
     */
    ADB prod_bhp(int table,
            const ADB& flo,
            const ADB& thp,
            const ADB& wfr,
            const ADB& gfr,
            const ADB& alq);

    //FIXME: ARB: Implement inj_bhp to match the prod_bhp's, but for injection wells.

    /**
     * Computes the flo parameter according to the flo_type_
     * @return Production rate of oil, gas or liquid.
     */
    static ADB getFlo(const ADB& aqua, const ADB& liquid, const ADB& vapour,
                      const VFPProdTable::FLO_TYPE& type);

    /**
     * Computes the wfr parameter according to the wfr_type_
     * @return Production rate of oil, gas or liquid.
     */
    static ADB getWFR(const ADB& aqua, const ADB& liquid, const ADB& vapour,
                      const VFPProdTable::WFR_TYPE& type);

    /**
     * Computes the gfr parameter according to the gfr_type_
     * @return Production rate of oil, gas or liquid.
     */
    static ADB getGFR(const ADB& aqua, const ADB& liquid, const ADB& vapour,
                      const VFPProdTable::GFR_TYPE& type);

private:
    // Map which connects the table number with the table itself
    std::map<int, const VFPProdTable*> m_prod_tables;
    std::map<int, const VFPInjTable*> m_inj_tables;

    /**
     * Helper struct for linear interpolation
     */
    struct InterpData {
        InterpData() : ind_{0, 0}, factor_(0.0) {}
        int ind_[2]; //[First element greater than or equal to value, Last element smaller than or equal to value]
        double factor_; //Interpolation factor
    };

    /**
     * Helper function to find indices etc. for linear interpolation
     */
    static InterpData find_interp_data(const double& value, const std::vector<double>& values);

    /**
     * Helper function which interpolates data using the indices etc. given in the inputs.
     */
    static double interpolate(const VFPProdTable::array_type& array,
            const InterpData& flo_i,
            const InterpData& thp_i,
            const InterpData& wfr_i,
            const InterpData& gfr_i,
            const InterpData& alq_i);

    /**
     * Initialization routines
     */
    void init(const std::map<int, VFPInjTable>& inj_tables);
    void init(const std::map<int, VFPProdTable>& prod_tables);
};

}

#endif /* OPM_AUTODIFF_VFPPROPERTIES_HPP_ */
