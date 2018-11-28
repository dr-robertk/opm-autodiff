/*
  Copyright 2016 IRIS AS

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

#ifndef OPM_MATRIX_BLOCK_HEADER_INCLUDED
#define OPM_MATRIX_BLOCK_HEADER_INCLUDED

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/version.hh>
#include <dune/istl/matrixutils.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/superlu.hh>

#include <ewoms/linear/matrixblock.hh>

namespace Dune
{

template<class K, int n, int m>
using MatrixBlock = Ewoms::MatrixBlock< K, n, m >;

} // end namespace Dune

namespace Opm
{
namespace Detail
{
    //! calculates ret = A^T * B
    template< class K, int m, int n, int p >
    static inline void multMatrixTransposed ( const Dune::FieldMatrix< K, n, m > &A,
                                              const Dune::FieldMatrix< K, n, p > &B,
                                              Dune::FieldMatrix< K, m, p > &ret )
    {
        typedef typename Dune::FieldMatrix< K, m, p > :: size_type size_type;

        for( size_type i = 0; i < m; ++i )
        {
            for( size_type j = 0; j < p; ++j )
            {
                ret[ i ][ j ] = K( 0 );
                for( size_type k = 0; k < n; ++k )
                    ret[ i ][ j ] += A[ k ][ i ] * B[ k ][ j ];
            }
        }
    }
} // namespace Detail
} // namespace Opm

#endif
