/*
  Copyright 2013 Statoil ASA.

  This file is part of the Open Porous Media Project (OPM).

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

#ifndef OPM_PRINTCURVE_HPP_HEADER
#define OPM_PRINTCURVE_HPP_HEADER

#include <iostream>

namespace Opm
{

    template <class UnstructuredGrid>
    inline void printCurve( const UnstructuredGrid& grid, std::ostream& out )
    {
        const int numCells = grid.num_cells;
        const int dim = grid.dimensions;
        for( int cell = 0; cell<numCells; ++cell )
        {
            for ( int d=0; d<dim; ++d )
                out << grid.cell_centroids[ cell ][ d ] << " ";
            out << std::endl;
        }
    }
}

#endif  /* OPM_PRINTCURVE_HPP_HEADER */
